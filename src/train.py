from audioop import mul
import os
from tracemalloc import stop
import jittor as jt
from jittor import nn
from jittor import transform
from jittor.dataset import Dataset
from jittor.lr_scheduler import CosineAnnealingLR, MultiStepLR
# import sys
# import argparse
import numpy as np
# from PIL import Image
# from tqdm import tqdm
from mlp_mixer import MLPMixerForImageClassification
from conv_mixer import ConvMixer
from vision_transformer import VisionTransformer
from jittor.models.resnet import Resnet50
import shutil
from tqdm import tqdm
from jittor import Module
import pdb
import random
import math
from PIL import Image
from tensorboardX import SummaryWriter
writer = SummaryWriter()
# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
# import neptune.new as neptune
# import neptune.new as neptune
# import optuna
# import neptune.new.integrations.optuna as optuna_utils

# run = neptune.init(
#     project="fanxiaoyu1234/OxfordFlowers",
#     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsIm\
#                 FwaV9rZXkiOiJhNTEyOTJhYi0zN2E2LTQzMWQtODI3ZC1iNWFjY2M2NDdjMmUifQ==",
#     mode="offline"
# )  # your credentials
# neptune_callback = optuna_utils.NeptuneCallback(run)

jt.flags.use_cuda=1

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
    losses = []
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
        losses.append(loss.data[0])
    scheduler.step()
    return round(total_acc/total_num,4),round(sum(losses)/len(losses),4)

def train_one_epoch(model, train_loader, criterion, optimizer, accum_iter, scheduler):
    model.train()
    total_acc = 0
    total_num = 0
    losses = []
    for i, (images, labels) in enumerate(train_loader):
        # images,labels=cutmix(images,labels,num_classes=1020)
        output = model(images)
        loss = criterion(output, labels)
        optimizer.backward(loss)
        if (i + 1) % accum_iter == 0 or i + 1 == len(train_loader):
            optimizer.step(loss)
        pred = np.argmax(output.data, axis=1)
        acc = np.sum(pred == labels.data)
        # acc = np.sum(pred == np.argmax(labels.data, axis=1))
        total_acc += acc
        total_num += labels.shape[0]
        losses.append(loss.data[0])
    scheduler.step()
    return round(total_acc/total_num,4),round(sum(losses)/len(losses),4)

def valid_one_epoch(model, val_loader, epoch):
    model.eval()
    total_acc = 0
    total_num = 0
    for i, (images, labels) in enumerate(val_loader):
        output = model(images)
        pred = np.argmax(output.data, axis=1)
        acc = np.sum(pred == labels.data)
        total_acc += acc
        total_num += labels.shape[0]
    acc = total_acc / total_num
    return round(acc,4)

def calculate_test_set_accuracy(model):
    jt.set_global_seed(648)  
    model.eval()

    # region Processing data 
    resizedImageSize = 256
    croppedImagesize=resizedImageSize-32
    data_transforms = {
        'test': transform.Compose([
            transform.Resize((croppedImagesize, croppedImagesize)),
            transform.ToTensor(),
            transform.ImageNormalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ])
    }
    batch_size=64
    data_dir = '../data'
    image_datasets = {x: jt.dataset.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['test']}
    test_loader = image_datasets['test'].set_attrs(batch_size=batch_size, shuffle=False)
    # endregion

    total_acc = 0
    total_num = 0
    pbar = tqdm(test_loader, desc="calculate_test_set_accuracy")
    for (images, labels) in pbar:        
        output = model(images)
        pred = np.argmax(output.data, axis=1)
        acc = np.sum(pred == labels.data)
        total_acc += acc
        total_num += labels.shape[0]
        pbar.set_description(f'acc={total_acc / total_num:.4f}')
    acc = total_acc / total_num
    print(round(acc,4))

def trial(model:nn.Module,modelName,learningRate,epochs,etaMin,savedName):
    jt.set_global_seed(648)  

    # region Processing data 
    resizedImageSize = 256
    croppedImagesize=resizedImageSize-32
    data_transforms = {
        'train': transform.Compose([
            transform.Resize((resizedImageSize,resizedImageSize)),
            transform.RandomCrop((croppedImagesize, croppedImagesize)),       # 从中心开始裁剪
            transform.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
            transform.RandomVerticalFlip(p=0.5),    # 随机垂直翻转
            transform.RandomRotation(90),
            transform.ToTensor(),
            transform.ImageNormalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])  # 均值，标准差
        ]),
        'valid': transform.Compose([
            transform.Resize((croppedImagesize, croppedImagesize)),
            transform.ToTensor(),
            transform.ImageNormalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ]),
        'test': transform.Compose([
            transform.Resize((croppedImagesize, croppedImagesize)),
            transform.ToTensor(),
            transform.ImageNormalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ])
    }
    batch_size = 64
    data_dir = '../data'
    image_datasets = {x: jt.dataset.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                    ['train', 'valid', 'test']}
    traindataset = image_datasets['train'].set_attrs(batch_size=batch_size, shuffle=True)
    validdataset = image_datasets['valid'].set_attrs(batch_size=batch_size, shuffle=False)
    # endregion

    # region model and optimizer
    criterion = SoftLabelCrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = nn.Adam(model.parameters(), lr=learningRate, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, 15, eta_min=etaMin)
    # endregion

    # region train and valid
    print("----------------- A new trial ---------------------")
    print("modelName",modelName,"learning rate:",learningRate,"epochs",epochs,"etaMin",etaMin,"savedName",savedName)
    maxBearableEpochs=50
    noProgressEpochs=0
    stopEpoch=0
    currentBestAccuracy=0.0
    currentBestEpoch=0
    for epoch in range(epochs):
        trainAccuracy,trainLoss=train_one_epoch(model, traindataset, criterion, optimizer, 1, scheduler)
        validAccuracy=valid_one_epoch(model, validdataset, epoch)
        print("epoch:",epoch,"validAccuracy:",validAccuracy,"trainAccuracy:",trainAccuracy,"delta:",round(validAccuracy-currentBestAccuracy,4))
        writer.add_scalars(modelName,{'trainAccuracy':trainAccuracy,'validAccuracy':validAccuracy,"trainLoss":trainLoss}, epoch)
        if validAccuracy > currentBestAccuracy:
            currentBestAccuracy=validAccuracy
            currentBestEpoch=epoch
            model.save("../model/"+modelName+"/"+savedName)
            noProgressEpochs=0
        else:
            noProgressEpochs+=1
            if noProgressEpochs>=maxBearableEpochs:
                stopEpoch=epoch
                break
        stopEpoch=epoch
    print("==========================================================================================")
    print("validAccuracy",currentBestAccuracy,"bestEpoch",currentBestEpoch,"stopEpoch",stopEpoch)
    print("==========================================================================================")
    # endregion

def pretrain(model:nn.Module,modelName,learningRate,epochs,etaMin,savedName):
    jt.set_global_seed(648)  

    # region Processing data 
    resizedImageSize = 256
    croppedImagesize=resizedImageSize-32
    data_transforms = {
        'pretrain': transform.Compose([
            transform.Resize((resizedImageSize,resizedImageSize)),
            transform.RandomCrop((croppedImagesize, croppedImagesize)),       # 从中心开始裁剪
            transform.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
            transform.RandomVerticalFlip(p=0.5),    # 随机垂直翻转
            transform.RandomRotation(90),
            transform.ToTensor(),
            transform.ImageNormalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])  # 均值，标准差
        ]),
        'valid': transform.Compose([
            transform.Resize((croppedImagesize, croppedImagesize)),
            transform.ToTensor(),
            transform.ImageNormalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ]),
        'test': transform.Compose([
            transform.Resize((croppedImagesize, croppedImagesize)),
            transform.ToTensor(),
            transform.ImageNormalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ])
    }
    batch_size = 64
    data_dir = '../data'
    image_datasets = {x: jt.dataset.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                    ['pretrain', 'valid', 'test']}
    traindataset = image_datasets['pretrain'].set_attrs(batch_size=batch_size, shuffle=True)
    # endregion
    
    # region hyper parameters
    criterion = SoftLabelCrossEntropyLoss()
    optimizer = nn.Adam(model.parameters(), lr=learningRate, weight_decay=1e-3)    
    scheduler = CosineAnnealingLR(optimizer, 15, eta_min=etaMin)
    # endregion

    # region pretrain
    print("----------------- A new trial ---------------------")
    print("modelName",modelName,"learning rate:",learningRate,"epochs",epochs,"etaMin",etaMin,"savedName",savedName)
    currentBestAccuracy=0.0
    for epoch in range(epochs):
        trainAccuracy,trainLoss=pretrain_one_epoch(model, traindataset, criterion, optimizer, 1, scheduler)
        print("epoch:",epoch,"trainAccuracy:",trainAccuracy,"delta:",round(trainAccuracy-currentBestAccuracy,4))
        writer.add_scalars(modelName,{'trainAccuracy':trainAccuracy,"trainLoss":trainLoss}, epoch)
        if trainAccuracy > currentBestAccuracy:
            currentBestAccuracy=trainAccuracy
        if epoch%20==0:    
            model.save("../model/"+modelName+"/"+savedName)
    # endregion

def finetune(model:nn.Module,modelName,learningRate,epochs,etaMin,savedName):
    jt.set_global_seed(648)  

    # region Processing data 
    resizedImageSize = 256
    croppedImagesize=resizedImageSize-32
    data_transforms = {
        'train': transform.Compose([
            transform.Resize((resizedImageSize,resizedImageSize)),
            transform.RandomCrop((croppedImagesize, croppedImagesize)),       # 从中心开始裁剪
            transform.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
            transform.RandomVerticalFlip(p=0.5),    # 随机垂直翻转
            transform.RandomRotation(90),
            transform.ToTensor(),
            transform.ImageNormalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])  # 均值，标准差
        ]),
        'valid': transform.Compose([
            transform.Resize((croppedImagesize, croppedImagesize)),
            transform.ToTensor(),
            transform.ImageNormalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ]),
        'test': transform.Compose([
            transform.Resize((croppedImagesize, croppedImagesize)),
            transform.ToTensor(),
            transform.ImageNormalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ])
    }
    batch_size = 64
    data_dir = '../data'
    image_datasets = {x: jt.dataset.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                    ['train', 'valid', 'test']}
    traindataset = image_datasets['train'].set_attrs(batch_size=batch_size, shuffle=True)
    validdataset = image_datasets['valid'].set_attrs(batch_size=batch_size, shuffle=False)
    # endregion

    # region model and optimizer
    criterion = SoftLabelCrossEntropyLoss()
    optimizer = nn.Adam(model.parameters(), lr=learningRate, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, 15, eta_min=etaMin)
    # endregion

    # region train and valid
    print("----------------- A new trial ---------------------")
    print("modelName",modelName,"learning rate:",learningRate,"epochs",epochs,"etaMin",etaMin,"savedName",savedName)
    maxBearableEpochs=50
    noProgressEpochs=0
    stopEpoch=0
    currentBestAccuracy=0.0
    currentBestEpoch=0
    for epoch in range(epochs):
        trainAccuracy,trainLoss=train_one_epoch(model, traindataset, criterion, optimizer, 1, scheduler)
        validAccuracy=valid_one_epoch(model, validdataset, epoch)
        print("epoch:",epoch,"validAccuracy:",validAccuracy,"trainAccuracy:",trainAccuracy,"delta:",round(validAccuracy-currentBestAccuracy,4))
        writer.add_scalars(modelName,{'trainAccuracy':trainAccuracy,'validAccuracy':validAccuracy,"trainLoss":trainLoss}, epoch)
        if validAccuracy > currentBestAccuracy:
            currentBestAccuracy=validAccuracy
            currentBestEpoch=epoch
            model.save("../model/"+modelName+"/"+savedName)
            noProgressEpochs=0
        else:
            noProgressEpochs+=1
            if noProgressEpochs>=maxBearableEpochs:
                stopEpoch=epoch
                break
        stopEpoch=epoch
    print("==========================================================================================")
    print("validAccuracy",currentBestAccuracy,"bestEpoch",currentBestEpoch,"stopEpoch",stopEpoch)
    print("==========================================================================================")
    # endregion

def multi_predict(model:nn.Module):
    jt.set_global_seed(648)  
    model.eval()

    resizedImageSize = 256
    croppedImagesize=resizedImageSize-32
    classNameList=os.listdir('../data/test/')
    classNameList.sort()
    totalPredictNumber=0
    rightPredictNumber=0
    pbar=tqdm(enumerate(classNameList),desc="multi_predict")
    for i,className in pbar:
        imageNameList=os.listdir('../data/test/'+className+"/")
        for imageName in imageNameList:
            tempImages=[]
            for j in range(64):
                image = Image.open('../data/test/'+className+"/"+imageName).convert('RGB')
                image=transform.Resize((resizedImageSize,resizedImageSize))(image)
                image=transform.RandomCrop((croppedImagesize, croppedImagesize))(image)
                image=transform.RandomHorizontalFlip(p=0.5)(image)
                image=transform.RandomVerticalFlip(p=0.5)(image)
                image=transform.RandomRotation(90)(image)
                image=transform.ToTensor()(image)
                image=transform.ImageNormalize([0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])(image)
                tempImages.append(np.array(image))
            pred:jt.Var=model(jt.Var(tempImages))
            pred = np.argmax(pred.sum(0))
            # print(pred)
            if pred==i:
                rightPredictNumber+=1
            totalPredictNumber+=1
        pbar.set_description(f'testAccuracy={rightPredictNumber/totalPredictNumber:.4f}')
    return round(rightPredictNumber/totalPredictNumber,4)

def divide_train_dataset():
    # group=0
    count=0
    classDirList=os.listdir("../data/train/")
    for classDir in tqdm(classDirList,desc="divide_train_dataset"):
        imageNameList=os.listdir("../data/train/"+classDir+"/")
        for imageName in imageNameList:
            # if count%102==0
            os.makedirs("../data/pretrain/"+str(count))
            shutil.copyfile("../data/train/"+classDir+"/"+imageName, "../data/pretrain/"+str(count)+"/"+imageName)
            count+=1

if __name__=="__main__":
    #======================== ViT ==============================
    # vitModel = VisionTransformer(img_size=224,patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3.,num_classes=1020)
    # # pretrain(model=vitModel,modelName="ViT",learningRate=5e-5,epochs=300,etaMin=1e-5,savedName="pretrained_4.pkl")
    # vitModel.load("../model/ViT/pretrained_2.pkl")
    # finetune(model=vitModel,modelName="ViT",learningRate=5e-5,epochs=200,etaMin=1e-5,savedName="finetuned_4_1.pkl")
    # # vitModel.load("../model/ViT/finetuned_4_1.pkl")
    # multi_predict(vitModel)

    #======================= MLPMixer =============================
    mlpMixerModel=MLPMixerForImageClassification(
        in_channels=3,patch_size=16, d_model=512, depth=12, num_classes=1020,image_size=224,dropout=0)
    print(mlpMixerModel)
    print(mlpMixerModel._modules.keys())
    fdhsjk
    # trial(model=mlpMixerModel,modelName="MLPMixer",learningRate=2.3e-5,epochs=200,etaMin=1e-5,savedName="MLPMixer_4.pkl")
    # pretrain(model=mlpMixerModel,modelName="MLPMixer",learningRate=2.3e-5,epochs=200,etaMin=1e-7,savedName="pretrained_2.pkl")
    # mlpMixerModel.load("../model/MLPMixer/pretrained_2.pkl")
    # finetune(model=mlpMixerModel,modelName="MLPMixer",learningRate=2.3e-5,epochs=200,etaMin=1e-7,savedName="finetuned_2.pkl")
    # mlpMixerModel.load("../model/MLPMixer/finetuned_2.pkl")
    # mlpMixerModel.load("../model/MLPMixer/MLPMixer_3.pkl")
    # calculate_test_set_accuracy(mlpMixerModel)
    # testAccuracy=multi_predict(mlpMixerModel)
    # print("testAccuracy",testAccuracy)

    #====================== ConvMixer =================================
    convMixerModel=ConvMixer(dim = 768, depth = 32, kernel_size=7, patch_size=7,n_classes=102)
    # trial(model=convMixerModel,modelName="ConvMixer",learningRate=1e-4,epochs=500,etaMin=1e-7,savedName="ConvMixer_2.pkl")
    # pretrain(model=convMixerModel,modelName="ConvMixer",learningRate=1e-4,epochs=200,etaMin=1e-7,savedName="pretrained_2.pkl")
    # convMixerModel.load("../model/ConvMixer/pretrained_1.pkl")
    convMixerModel.load("../model/ConvMixer/ConvMixer.pkl")
    # finetune(model=convMixerModel,modelName="ConvMixer",learningRate=1e-4,epochs=300,etaMin=1e-7,savedName="finetuned_1.pkl")
    # convMixerModel.load("../model/ConvMixer/finetuned_1.pkl")
    print(convMixerModel)
    print(convMixerModel._modules.keys())
    
    # calculate_test_set_accuracy(convMixerModel)
    # testAccuracy=multi_predict(convMixerModel)
    # print("testAccuracy",testAccuracy)

    #====================== ResNet =================================
    # resnetModel=Resnet50(num_classes=102)
    # # trial(model=convMixerModel,modelName="ConvMixer",learningRate=1e-4,epochs=500,etaMin=1e-8,savedName="ConvMixer_2.pkl")
    # # pretrain(model=convMixerModel,modelName="ConvMixer",learningRate=3e-3,epochs=800,etaMin=1e-8,savedName="pretrained_1.pkl")
    # # convMixerModel.load("../model/ConvMixer/pretrained_1.pkl")
    # # finetune(model=convMixerModel,modelName="ConvMixer",learningRate=3e-3,epochs=400,etaMin=1e-8,savedName="finetuned_1.pkl")
    # # convMixerModel.load("../model/ConvMixer/finetuned_1.pkl")
    # resnetModel.load("../model/ResNet/best.pkl")
    # calculate_test_set_accuracy(resnetModel)
    # # testAccuracy=multi_predict(convMixerModel)
    # # print("testAccuracy",testAccuracy)
    
    writer.close()
















    # convMixerModel.load("../model/ConvMixer/best.pkl")
    
    
    
    # calculate_test_set_accuracy(vitModel)
    

    # multicrop(vitModel)
    # trial(model=vitModel,modelName="ViT",learningRate=5e-4,epochs=800)
    # divide_train_dataset()
    # writer.close()
    # resnetModel=Resnet50(num_classes=102)
    # trial(model=resnetModel,modelName="ResNet",learningRate=1e-4,epochs=1600)



    # globalBestAccuracy=0.0
    # learningRateList=[1e-4]#[1e-6,1e-5,5e-5,1e-4,1e-3]
    # trialBoard=[]
    # for learningRate in learningRateList:
    #     accuracy,epoch=trial(learningRate=learningRate)
    #     trialBoard.append("learningRate: "+str(learningRate)+" accuracy:"+str(accuracy)+" epoch: "+str(epoch))
    #     print("=============================================================================================")
    #     print("learningRate: "+str(learningRate)+" accuracy:"+str(accuracy)+" epoch: "+str(epoch)+" delta: "+str(round(accuracy-globalBestAccuracy,3)))
    #     print("=============================================================================================")
    #     if accuracy>globalBestAccuracy:
    #         globalBestAccuracy=accuracy
    # for  oneTrial in trialBoard:
    #     print(oneTrial) 
    # Add stream handler of stdout to show the messages
    # optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    # study = optuna.create_study(direction="maximize",pruner=optuna.pruners.MedianPruner(n_warmup_steps=30))
    # study.optimize(objective, n_trials=1, callbacks=[neptune_callback])
    # best_params = study.best_params
    # print(best_params)
    # run.stop()

