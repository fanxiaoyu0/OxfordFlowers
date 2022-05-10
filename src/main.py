from audioop import mul
# from copyreg import pickle
import os
from tracemalloc import stop
import jittor as jt
from jittor import nn
from jittor import transform
from jittor.dataset import Dataset
from jittor.lr_scheduler import CosineAnnealingLR, MultiStepLR
import sys
# import argparse
import numpy as np
# from PIL import Image
from MLPMixer import MLPMixerForImageClassification
from ConvMixer import ConvMixer
from ViT import VisionTransformer
from jittor.models.resnet import Resnet50
import shutil
from tqdm import tqdm
from jittor import Module
import pdb
import random
import math
from PIL import Image
from tensorboardX import SummaryWriter
import time
import smtplib, ssl
from trycourier import Courier
import pickle

writer = SummaryWriter()
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
jt.flags.use_cuda=1
jt.set_global_seed(648)

def send_email_to_myself(stringData):
    client = Courier(auth_token="pk_prod_FRV8652N0ZMXW1KPF236PV7ADNV1")
    resp = client.send_message(
        message={
            "to": {
                "email": "fan-xy19@mails.tsinghua.edu.cn"
            },
            "content": {
                "title": stringData,
                "body": stringData
            },
            # "data":{
            #     "joke": "Why does Python live on land? Because it is above C level"
            # }
        }
    )

class SoftLabelCrossEntropyLoss(Module):
    def __init__(self):
        self.epsilon=0.1
        self.n=1020
        
    def execute(self, output:jt.Var, target:jt.Var):
        if target.ndim == 1:
            target = target.reshape((-1, ))
            target = target.broadcast(output, [1])
            target = target.index(1) == target
        target_weight = jt.ones(target.shape[0], dtype='float32')
        softTarget=target*(1-self.epsilon-self.epsilon/(self.n-1))+self.epsilon/(self.n-1)
        output = output - output.max([1], keepdims=True)
        logsum = output.exp().sum(1).log()
        loss:jt.Var = (logsum - (output*softTarget).sum(1)) * target_weight
        return loss.sum() / target_weight.sum()

def cutmix(batch:jt.Var, target:jt.Var,num_classes,p=0.5,alpha=1.0):
    if target.ndim == 1:
        target = np.eye(num_classes, dtype=np.float32)[target]
    if random.random() >= p:
        return batch, target
        
    batch_rolled = np.roll(batch, 1, 0)
    target_rolled = np.roll(target, 1,0)
    
    lambda_param =np.random.beta(alpha, alpha)
    W, H = batch.shape[-1], batch.shape[-2]
    # print(W,H)
    # pdb.set_trace()
    
    r_x = np.random.randint(W)
    r_y = np.random.randint(H)
    
    r = 0.5 * math.sqrt(1. - lambda_param)
    r_w_half = int(r * W)
    r_h_half = int(r * H)
    
    x1 = int(max(r_x - r_w_half, 0))
    y1 = int(max(r_y - r_h_half, 0))
    x2 = int(min(r_x + r_w_half, W))
    y2 = int(min(r_y + r_h_half, H))
    
    batch[:,:,y1:y2,x1:x2] = batch_rolled[:,:,y1:y2,x1:x2]
    lambda_param = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    
    target_rolled *= 1 - lambda_param
    target = target * lambda_param + target_rolled
    target=jt.float32(target)
    # print(type(target))
    # print(target[0][0:100])
    # pdb.set_trace()
    return batch, target

class RandomCutmix:
    def __init__(self, num_classes, p=0.5, alpha=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        
    def __call__(self, batch, target:jt.Var):
        if target.ndim == 1:
            target = np.eye(self.num_classes, dtype=np.float32)[target]
            
        if random.random() >= self.p:
            return batch, target
            
        batch_rolled = np.roll(batch, 1, 0)
        target_rolled = np.roll(target, 1,0)
        
        lambda_param =0.5# np.random.beta(self.alpha, self.alpha)
        W, H = batch.shape[-1], batch.shape[-2]
        
        r_x = np.random.randint(W)
        r_y = np.random.randint(H)
        
        r = 0.5 * math.sqrt(1. - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)
        
        x1 = int(max(r_x - r_w_half, 0))
        y1 = int(max(r_y - r_h_half, 0))
        x2 = int(min(r_x + r_w_half, W))
        y2 = int(min(r_y + r_h_half, H))
        
        batch[:,:,y1:y2,x1:x2] = batch_rolled[:,:,y1:y2,x1:x2]
        lambda_param = 1 - (x2 - x1) * (y2 - y1) / (W * H)
        
        target_rolled *= 1 - lambda_param
        target = target * lambda_param + target_rolled
        
        return batch, target

def pretrain_one_epoch(model:nn.Module, train_loader, criterion, optimizer, accum_iter, scheduler):
    model.train()
    total_acc = 0
    total_num = 0
    lossList = []
    for i, (images, labels) in enumerate(train_loader):
        images,labels=cutmix(images,labels,num_classes=1020)
        output = model(images)
        loss = criterion(output, labels)
        optimizer.backward(loss)
        if (i + 1) % accum_iter == 0 or i + 1 == len(train_loader):
            optimizer.step(loss)
        pred = np.argmax(output.data, axis=1)
        acc = np.sum(pred == np.argmax(labels.data, axis=1))
        total_acc += acc
        total_num += labels.shape[0]
        lossList.append(loss.data[0])
    scheduler.step()
    return round(total_acc/total_num,4),round(sum(lossList)/len(lossList),4)

def train_one_epoch(model:nn.Module, trainDataSet, criterion, optimizer, scheduler):
    model.train()
    rightCount = 0
    totalCount = 0
    lossList = []
    # for index, (images, labels) in tqdm(enumerate(trainDataSet)):
    for index, (images, labels) in enumerate(trainDataSet):
        output = model(images)
        loss = criterion(output, labels)
        optimizer.backward(loss)
        optimizer.step(loss)
        predictLabel = np.argmax(output.data, axis=1)
        rightCount += np.sum(predictLabel == labels.data)
        totalCount += labels.shape[0]
        lossList.append(loss.data[0])
    scheduler.step()
    return round(rightCount/totalCount,4),round(sum(lossList)/len(lossList),4)

def validate_one_epoch(model:nn.Module, validateDataSet,criterion):
    model.eval()
    rightCount = 0
    totalCount = 0
    lossList = []
    for index, (images, labels) in enumerate(validateDataSet):
        output = model(images)
        loss = criterion(output, labels)
        predictLabel = np.argmax(output.data, axis=1)
        rightCount += np.sum(predictLabel == labels.data)
        totalCount += labels.shape[0]
        lossList.append(loss.data[0])
    return round(rightCount/totalCount,4),round(sum(lossList)/len(lossList),4)

def get_three_data_set_accuracy(model,imageSize,batchSize):
    trainDataSet, validateDataSet, testDataSet=construct_data_loader(imageSize=imageSize,batchSize=batchSize)
    model.eval()
    print("Calculating train data set accuracy...")
    rightCount = 0
    totalCount = 0
    for index, (images, labels) in enumerate(trainDataSet):
        output = model(images)
        predictLabel = np.argmax(output.data, axis=1)
        rightCount += np.sum(predictLabel == labels.data)
        totalCount += labels.shape[0]
    trainAccuracy=round(rightCount/totalCount,4)
    print("Train data set accuracy:",trainAccuracy)
    print("Calculating validate data set accuracy...")
    rightCount = 0
    totalCount = 0
    for index, (images, labels) in enumerate(validateDataSet):
        output = model(images)
        predictLabel = np.argmax(output.data, axis=1)
        rightCount += np.sum(predictLabel == labels.data)
        totalCount += labels.shape[0]
    validateAccuracy=round(rightCount/totalCount,4)
    print("Validate data set accuracy:",validateAccuracy)
    print("Calculating test data set accuracy...")
    rightCount = 0
    totalCount = 0
    for index, (images, labels) in enumerate(testDataSet):
        output = model(images)
        predictLabel = np.argmax(output.data, axis=1)
        rightCount += np.sum(predictLabel == labels.data)
        totalCount += labels.shape[0]
    testAccuracy=round(rightCount/totalCount,4)
    print("Test data set accuracy:",testAccuracy)
    return trainAccuracy,validateAccuracy,testAccuracy 

def construct_data_loader(imageSize,batchSize):
    resizedImageSize=imageSize+32
    data_transforms = {
        'train': transform.Compose([
            transform.Resize((resizedImageSize,resizedImageSize)),
            transform.RandomCrop((imageSize, imageSize)),       # 从中心开始裁剪
            transform.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
            transform.RandomVerticalFlip(p=0.5),    # 随机垂直翻转
            transform.RandomRotation(90),
            transform.ToTensor(),
            transform.ImageNormalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])  # 均值，标准差
        ]),
        'validate': transform.Compose([
            transform.Resize((imageSize, imageSize)),
            transform.ToTensor(),
            transform.ImageNormalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ]),
        'test': transform.Compose([
            transform.Resize((imageSize, imageSize)),
            transform.ToTensor(),
            transform.ImageNormalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ])
    }
    print("Loading data...")
    data_dir = '../data'
    image_datasets = {x: jt.dataset.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'validate', 'test']}
    trainDataSet = image_datasets['train'].set_attrs(batch_size=batchSize, shuffle=True)
    validateDataSet = image_datasets['validate'].set_attrs(batch_size=batchSize, shuffle=False)
    testDataSet = image_datasets['test'].set_attrs(batch_size=batchSize, shuffle=False)
    return trainDataSet, validateDataSet, testDataSet

def train(model:nn.Module,modelName,learningRate,epochs,etaMin,imageSize,batchSize,savedName):
    trainDataSet, validateDataSet, testDataSet=construct_data_loader(imageSize,batchSize) 

    # criterion = SoftLabelCrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = nn.Adam(model.parameters(), lr=learningRate, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, 15, eta_min=etaMin)

    print("----------------- A new trial ---------------------")
    print("modelName",modelName,"learning rate:",learningRate,"etaMin",etaMin,"imgSize",imageSize,"batchSize",batchSize,"savedName",savedName,\
        "criterion","CrossEntropyLoss","weight_decay",1e-3,"TMax",15)
    with open("../result/summary/"+modelName+".txt","a") as f:
        f.write("modelName: "+modelName+"  learning rate:"+str(learningRate)+"  etaMin:"+str(etaMin)+"  imgSize:"+str(imageSize)+\
            "  batchSize:"+str(batchSize)+"  savedName:"+savedName+"  criterion:"+"  CrossEntropyLoss"+"  weight_decay:"+str(1e-3)+"  TMax:"+str(15)+"\n")
    maxBearableEpochs=50
    noProgressEpochs=0
    stopEpoch=0
    currentBestAccuracy=0.0
    currentBestEpoch=0
    for epoch in tqdm(range(epochs),desc="Training"):
        trainAccuracy,trainLoss=train_one_epoch(model, trainDataSet, criterion, optimizer, scheduler)
        validateAccuracy,validateLoss=validate_one_epoch(model, validateDataSet,criterion)
        print("epoch:",epoch,"validateAccuracy:",validateAccuracy,"trainAccuracy:",trainAccuracy,"delta:",round(validateAccuracy-currentBestAccuracy,4))
        writer.add_scalars(modelName,{'trainAccuracy':trainAccuracy,'validateAccuracy':validateAccuracy,"trainLoss":trainLoss,"validateLoss":validateLoss}, epoch)
        if validateAccuracy > currentBestAccuracy:
            currentBestAccuracy=validateAccuracy
            currentBestEpoch=epoch
            model.save("../weight/"+modelName+"/"+savedName)
            noProgressEpochs=0
        else:
            noProgressEpochs+=1
            if noProgressEpochs>=maxBearableEpochs:
                stopEpoch=epoch
                break
        stopEpoch=epoch
    model.load("../weight/"+modelName+"/"+savedName)
    trainAccuracy,validateAccuracy,testAccuracy=get_three_data_set_accuracy(model,imageSize,batchSize)
    print("==========================================================================================")
    print("trainAccuracy",trainAccuracy,"validateAccuracy",validateAccuracy,"testAccuracy",testAccuracy,\
        "bestEpoch",currentBestEpoch,"stopEpoch",stopEpoch)
    print("==========================================================================================")
    send_email_to_myself(modelName+" Training Completed!")

if __name__=="__main__":
    # vitPretrainedModel = VisionTransformer(img_size=224,patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3.,num_classes=5001)
    # vitModel.load("../../FSPT/weight/ViT/0.pkl")
    # vitPretrainedModel=pickle.load(open("../../FSPT/weight/ViT/1.pkl","rb"))
    # vitModel = VisionTransformer(img_size=224,patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3.,num_classes=102)
    # vitModel.save("../weight/ViT/temp.pkl")
    # vitModel=pickle.load(open("../weight/ViT/temp.pkl","rb"))
    # for key in vitModel.keys():
    #     if 'head' not in key:
    #         vitModel[key]=vitPretrainedModel[key]
    # pickle.dump(vitModel,open("../weight/ViT/1_2.pkl","wb"))
    # vitModel = VisionTransformer(img_size=224,patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3.,num_classes=102)
    # vitModel.load("../weight/ViT/1_2.pkl")
    # train(model=vitModel,modelName="ViT",learningRate=5e-5,epochs=200,etaMin=1e-5,imageSize=224,batchSize=16,savedName="1_2.pkl",)

    # vitModel=VisionTransformer(img_size=224,patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3.,num_classes=102)
    # # vitModel=pickle.load(open("../weight/ViT/0_2.pkl","rb"))
    # # print(vitModel.keys())
    # # print(vitModel['head.weight'].shape)
    # vitModel.load("../weight/ViT/0_2.pkl")
    # trainAccuracy,validateAccuracy,testAccuracy=get_three_data_set_accuracy(model=vitModel,imageSize=224,batchSize=64)
    # print("trainAccuracy",trainAccuracy,"validateAccuracy",validateAccuracy,"testAccuracy",testAccuracy)

    # mlpMixerModel=MLPMixerForImageClassification(
    #     in_channels=3,patch_size=16, d_model=512, depth=12, num_classes=102,image_size=224,dropout=0)
    # # mlpMixerModel.load("../../FSPT/weight/MLPMixer/1.pkl")
    mlpMixerPretrainedModel=pickle.load(open("../../FSPT/weight/MLPMixer/1.pkl","rb"))
    mlpMixerModel = MLPMixerForImageClassification(
        in_channels=3,patch_size=16, d_model=512, depth=12, num_classes=102,image_size=224,dropout=0)
    mlpMixerModel.save("../weight/MLPMixer/temp.pkl")
    mlpMixerModel=pickle.load(open("../weight/MLPMixer/temp.pkl","rb"))
    for key in mlpMixerModel.keys():
        if 'head' not in key:
            mlpMixerModel[key]=mlpMixerPretrainedModel[key]
    pickle.dump(mlpMixerModel,open("../weight/MLPMixer/1_2.pkl","wb"))
    mlpMixerModel = MLPMixerForImageClassification(
        in_channels=3,patch_size=16, d_model=512, depth=12, num_classes=102,image_size=224,dropout=0)
    mlpMixerModel.load("../weight/MLPMixer/1_2.pkl")
    train(model=mlpMixerModel,modelName="MLPMixer",learningRate=2.3e-5,epochs=200,etaMin=1e-5,imageSize=224,batchSize=64,savedName="1_2.pkl")

    # mlpMixerModel=MLPMixerForImageClassification(
    #     in_channels=3,patch_size=16, d_model=512, depth=12, num_classes=102,image_size=224,dropout=0)
    # mlpMixerModel.load("../weight/MLPMixer/0.pkl")
    # trainAccuracy,validateAccuracy,testAccuracy=get_three_data_set_accuracy(model=mlpMixerModel,imageSize=224,batchSize=64)
    # print("trainAccuracy",trainAccuracy,"validateAccuracy",validateAccuracy,"testAccuracy",testAccuracy)
    
    # convMixerModel=ConvMixer(dim = 768, depth = 32, kernel_size=7, patch_size=7,n_classes=102)
    # train(model=convMixerModel,modelName="ConvMixer",learningRate=1e-4,epochs=500,etaMin=1e-7,imageSize=224,batchSize=32,savedName="0_1.pkl")

    # convMixerModel=ConvMixer(dim = 768, depth = 32, kernel_size=7, patch_size=7,n_classes=102)
    # convMixerModel.load("../weight/ConvMixer/0.pkl")
    # trainAccuracy,validateAccuracy,testAccuracy=get_three_data_set_accuracy(model=convMixerModel,imageSize=224,batchSize=64)
    # print("trainAccuracy",trainAccuracy,"validateAccuracy",validateAccuracy,"testAccuracy",testAccuracy)

    # resnetModel=Resnet50(num_classes=102)
    # train(model=resnetModel,modelName="ResNet",learningRate=1e-4,epochs=1600,etaMin=1e-7,imageSize=224,batchSize=64,savedName="0_1.pkl")

    # resnetModel=Resnet50(num_classes=102)
    # resnetModel.load("../weight/ResNet/0.pkl")
    # trainAccuracy,validateAccuracy,testAccuracy=get_three_data_set_accuracy(model=resnetModel,imageSize=224,batchSize=64)
    # print("trainAccuracy",trainAccuracy,"validateAccuracy",validateAccuracy,"testAccuracy",testAccuracy)

    writer.close()
