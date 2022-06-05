import os
import numpy as np
from tqdm import tqdm
import random
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image 

import torch
from torch import nn
from torchinfo import summary
from torchvision.io import read_image
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as T
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from ViT import vit_b_16,GaborViT

torch.random.manual_seed(1024)
torch.cuda.manual_seed(1024)
random.seed(1024)
np.random.seed(1024)
writer=SummaryWriter()

class MyDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.transform = transform
        self.image_path_list=[]
        self.label_list=[]
        # self.image_list=[]
        class_list=os.listdir(image_dir)
        for image_class in class_list:
            image_name_list=os.listdir(image_dir+'/'+image_class)
            for image_name in image_name_list:
                image_path=image_dir+'/'+image_class+'/'+image_name
                self.image_path_list.append(image_path)
                # image=Image.open(image_path)
                # image=T.Resize((512,512))(image)
                # image=T.ToTensor()(image)
                # self.image_list.append(image)
                self.label_list.append(int(image_class))
        # self.image_list=torch.stack(self.image_list).to('cuda')
        # self.label_list=torch.tensor(self.label_list).to('cuda')
                
    def __len__(self):
        return len(self.label_list)
    def __getitem__(self, index):
        image=Image.open(self.image_path_list[index])
        image=T.ToTensor()(image).to('cuda')
        # print(image.device)
        # fsdhjk
        # image = read_image(self.image_path_list[index]).to('cuda')
        # print(image)

        label = self.label_list[index]
        # image=self.image_list[index]
        image = self.transform(image)
        # image=image
        # print(image)
        return image, torch.tensor(label,).to('cuda')

def construct_data_loader(batch_size):
    print("Constructing data loader ...")
    # train_transform=T.AutoAugment(T.AutoAugmentPolicy.CIFAR10)
    # print(type(train_transform))

    train_transform=T.Compose([
        T.RandomResizedCrop(size=(224, 224),scale=(0.64,1)),
        T.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
        T.RandomVerticalFlip(p=0.5),    # 随机垂直翻转
        T.RandomRotation((-90,90)),
        # T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])  # 均值，标准差
    ])
    # print(type(train_transform))
    # print(type(T.ToTensor()))
    # dsfhj
    validate_transform=T.Compose([
        T.Resize(size=(224, 224)),
        # T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    test_transform=T.Compose([
        T.Resize(size=(224, 224)),
        # T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    train_dataset=MyDataset(image_dir='../data/train',transform=train_transform)
    validate_dataset=MyDataset(image_dir='../data/validate',transform=validate_transform)
    test_dataset=MyDataset(image_dir='../data/test',transform=test_transform)

    train_data_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    validate_data_loader=DataLoader(validate_dataset,batch_size=batch_size,shuffle=True)
    test_data_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True)
    return train_data_loader,validate_data_loader,test_data_loader

def train_one_epoch(model:nn.Module, trainDataSet, criterion, optimizer, scheduler):
    model.train()
    rightCount = 0
    totalCount = 0
    lossList = []
    # for index, (images, labels) in tqdm(enumerate(trainDataSet)):
    for (images, labels) in trainDataSet:
        # print(images.device)
        # images.to('cuda')
        # print(type(images))
        # print(images.device)
        # fdshkj
        output = model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictLabel = torch.argmax(output, axis=1)
        rightCount += torch.sum(predictLabel == labels)
        totalCount += labels.shape[0]
        lossList.append(loss.item())
    scheduler.step()
    return round((rightCount/totalCount).item(),4),round(sum(lossList)/len(lossList),4)

def validate_one_epoch(model:nn.Module, validateDataSet,criterion):
    model.eval()
    rightCount = 0
    totalCount = 0
    lossList = []
    # for (images, labels) in tqdm(validateDataSet):
    for (images, labels) in validateDataSet:
        output = model(images)
        loss = criterion(output, labels)
        predictLabel = torch.argmax(output, axis=1)
        rightCount += torch.sum(predictLabel == labels)
        totalCount += labels.shape[0]
        lossList.append(loss.item())
    return round((rightCount/totalCount).item(),4),round(sum(lossList)/len(lossList),4)

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

def trial(model:nn.Module,modelName,learningRate,epochs,etaMin,batchSize,savedName):
    trainDataSet, validateDataSet, testDataSet=construct_data_loader(batchSize) 

    # criterion = SoftLabelCrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=etaMin)

    print("----------------- A new trial ---------------------")
    print("modelName",modelName,"learning rate:",learningRate,"etaMin",etaMin,"batchSize",batchSize,"savedName",savedName,\
        "criterion","CrossEntropyLoss","weight_decay",1e-3,"TMax",15)
    # with open("../result/summary/"+modelName+".txt","a") as f:
        # f.write("modelName: "+modelName+"  learning rate:"+str(learningRate)+"  etaMin:"+str(etaMin)+"  imgSize:"+str(imageSize)+\
            # "  batchSize:"+str(batchSize)+"  savedName:"+savedName+"  criterion:"+"  CrossEntropyLoss"+"  weight_decay:"+str(1e-3)+"  TMax:"+str(15)+"\n")
    maxBearableEpochs=30
    currentBestAccuracy=0.0
    currentBestEpoch=0
    for epoch in range(epochs):
        trainAccuracy,trainLoss=train_one_epoch(model, trainDataSet, criterion, optimizer, scheduler)
        validateAccuracy,validateLoss=validate_one_epoch(model, validateDataSet,criterion)
        print("epoch:",epoch,"validateAccuracy:",validateAccuracy,"trainAccuracy:",trainAccuracy,"delta:",round(validateAccuracy-currentBestAccuracy,4))
        writer.add_scalars(modelName,{'trainAccuracy':trainAccuracy,'validateAccuracy':validateAccuracy,"trainLoss":trainLoss,"validateLoss":validateLoss}, epoch)
        if validateAccuracy > currentBestAccuracy:
            currentBestAccuracy=validateAccuracy
            currentBestEpoch=epoch
            torch.save(model,"../weight/"+modelName+"/"+savedName)
        else:
            if epoch-currentBestEpoch>=maxBearableEpochs:
                break
    # model.load("../weight/"+modelName+"/"+savedName)
    # trainAccuracy,validateAccuracy,testAccuracy=get_three_data_set_accuracy(model,imageSize,batchSize)
    # print("==========================================================================================")
    # print("trainAccuracy",trainAccuracy,"validateAccuracy",validateAccuracy,"testAccuracy",testAccuracy,\
        # "bestEpoch",currentBestEpoch,"stopEpoch",stopEpoch)
    # print("==========================================================================================")
    # send_email_to_myself(modelName+" Training Completed!")

if __name__=="__main__":
    # vitModel=vit_b_16(image_size=224,num_classes=102)
    # vitModel.to('cuda')
    # trial(model=vitModel,modelName="ViT",learningRate=2e-5,epochs=200,etaMin=1e-6,batchSize=12,savedName="0_6.pkl",)
    # fsdhjk

    gabor_vit=GaborViT()
    gabor_vit.to('cuda')
    trial(model=gabor_vit,modelName="ViT",learningRate=5e-5,epochs=200,etaMin=1e-6,batchSize=12,savedName="0_7.pkl",)
    vfghj
    # summary(vitModel,input_size=(1,3,224,224))
    # fdshfkj
    # # vitModel=pickle.load(open("../weight/ViT/0_2.pkl","rb"))
    # # print(vitModel.keys())
    # # print(vitModel['head.weight'].shape)
    # vitModel.load("../weight/ViT/0_2.pkl")
    # trainAccuracy,validateAccuracy,testAccuracy=get_three_data_set_accuracy(model=vitModel,imageSize=224,batchSize=64)
    # print("trainAccuracy",trainAccuracy,"validateAccuracy",validateAccuracy,"testAccuracy",testAccuracy)

    # construct_data_loader(batch_size=64)
    # fsdhk
    # vitPretrainedModel = VisionTransformer(img_size=224,patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3.,num_classes=5001)
    # vitPretrainedModel.head.eval()
    # fhdk
    # vitModel.load("../../FSPT/weight/ViT/0.pkl")

    vitPretrainedModel=pickle.load(open("../../FSPT/weight/ViT/1.pkl","rb"))
    vitModel = VisionTransformer(img_size=224,patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3.,num_classes=102)
    vitModel.save("../weight/ViT/temp.pkl")
    vitModel=pickle.load(open("../weight/ViT/temp.pkl","rb"))
    # print(vitModel.keys())

    # fdshjs
    layerNameList=['blocks.4.norm1.weight', 'blocks.4.norm1.bias', 'blocks.4.attn.qkv.weight', 'blocks.4.attn.proj.weight', 'blocks.4.attn.proj.bias',\
        'blocks.4.norm2.weight', 'blocks.4.norm2.bias', 'blocks.4.mlp.fc1.weight', 'blocks.4.mlp.fc1.bias', 'blocks.4.mlp.fc2.weight', \
        'blocks.4.mlp.fc2.bias', 'blocks.5.norm1.weight', 'blocks.5.norm1.bias', 'blocks.5.attn.qkv.weight', 'blocks.5.attn.proj.weight', \
        'blocks.5.attn.proj.bias', 'blocks.5.norm2.weight', 'blocks.5.norm2.bias', 'blocks.5.mlp.fc1.weight', 'blocks.5.mlp.fc1.bias', \
        'blocks.5.mlp.fc2.weight', 'blocks.5.mlp.fc2.bias', 'blocks.6.norm1.weight', 'blocks.6.norm1.bias', 'blocks.6.attn.qkv.weight', \
        'blocks.6.attn.proj.weight', 'blocks.6.attn.proj.bias', 'blocks.6.norm2.weight', 'blocks.6.norm2.bias', 'blocks.6.mlp.fc1.weight', \
        'blocks.6.mlp.fc1.bias', 'blocks.6.mlp.fc2.weight', 'blocks.6.mlp.fc2.bias', 'blocks.7.norm1.weight', 'blocks.7.norm1.bias', \
        'blocks.7.attn.qkv.weight', 'blocks.7.attn.proj.weight', 'blocks.7.attn.proj.bias', 'blocks.7.norm2.weight', 'blocks.7.norm2.bias', \
        'blocks.7.mlp.fc1.weight', 'blocks.7.mlp.fc1.bias', 'blocks.7.mlp.fc2.weight', 'blocks.7.mlp.fc2.bias', 'norm.weight', 'norm.bias', \
        'head.weight', 'head.bias']
    for key in vitModel.keys():
        if key not in layerNameList:
            # print(key)
            vitModel[key]=vitPretrainedModel[key]
    pickle.dump(vitModel,open("../weight/ViT/1_3.pkl","wb"))
    vitModel = VisionTransformer(img_size=224,patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3.,num_classes=102)
    vitModel.load("../weight/ViT/1_3.pkl")
    # vitModel.__module__
    # print(vitModel.named_modules())
    # print(vitModel.modules()[0])
    # for i in range(len(vitModel.modules())):
        # print(i,vitModel.modules()[i])
    # print(vitModel.modules())
    # fdshkj
    train(model=vitModel,modelName="ViT",learningRate=5e-5,epochs=200,etaMin=1e-5,imageSize=224,batchSize=16,savedName="1_3.pkl",)

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

    # mlpMixerPretrainedModel=pickle.load(open("../../FSPT/weight/MLPMixer/1.pkl","rb"))
    # mlpMixerModel = MLPMixerForImageClassification(
    #     in_channels=3,patch_size=16, d_model=512, depth=12, num_classes=102,image_size=224,dropout=0)
    # mlpMixerModel.save("../weight/MLPMixer/temp.pkl")
    # mlpMixerModel=pickle.load(open("../weight/MLPMixer/temp.pkl","rb"))
    # for key in mlpMixerModel.keys():
    #     if 'head' not in key:
    #         mlpMixerModel[key]=mlpMixerPretrainedModel[key]
    # pickle.dump(mlpMixerModel,open("../weight/MLPMixer/1_2.pkl","wb"))
    # mlpMixerModel = MLPMixerForImageClassification(
    #     in_channels=3,patch_size=16, d_model=512, depth=12, num_classes=102,image_size=224,dropout=0)
    # mlpMixerModel.load("../weight/MLPMixer/1_2.pkl")
    # train(model=mlpMixerModel,modelName="MLPMixer",learningRate=2.3e-5,epochs=200,etaMin=1e-5,imageSize=224,batchSize=64,savedName="1_2.pkl")

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
