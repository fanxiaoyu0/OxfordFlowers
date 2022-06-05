import sys
import argparse
import numpy as np
from PIL import Image           
from tqdm import tqdm       
import datetime    
import json
import os
import shutil
import random
import matplotlib.pyplot as plt


random.seed(0)

def create_dataset():
    classDirList=os.listdir("../../../data/")
    for classDir in tqdm(classDirList,desc="divide_dataset"):
        if not os.path.exists("../data/train/"+classDir):
            os.makedirs("../data/train/"+classDir)
        if not os.path.exists("../data/valid/"+classDir):
            os.makedirs("../data/valid/"+classDir)
        if not os.path.exists("../data/test/"+classDir):
            os.makedirs("../data/test/"+classDir)
        imageNameList=os.listdir("../../../data/"+classDir+"/")
        random.shuffle(imageNameList)
        for imageName in imageNameList[0:12]:
            shutil.copyfile("../../../data/"+classDir+"/"+imageName, "../data/train/"+classDir+"/"+imageName)
        for imageName in imageNameList[12:14]:
            shutil.copyfile("../../../data/"+classDir+"/"+imageName, "../data/valid/"+classDir+"/"+imageName)
        for imageName in imageNameList[14:16]:
            shutil.copyfile("../../../data/"+classDir+"/"+imageName, "../data/test/"+classDir+"/"+imageName)

def draw_accuracy_curve():
    epochList=[i for i in range(279)]
    print(epochList)
    trainAccuracyList=[]
    validAccuracyList=[]
    with open("ConvMixerBest.txt") as f:
        lines=f.readlines()
        for line in lines:
            wordList=line.split(" ")
            validAccuracyList.append(float(wordList[3]))
            trainAccuracyList.append(float(wordList[5]))
    print(trainAccuracyList)
    print(validAccuracyList)
    plt.plot( epochList, trainAccuracyList, 'r')
    plt.plot( epochList, validAccuracyList, 'g')
    plt.savefig("ConvMixerBest.png")

def multi_predict(model:nn.Module):
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

def run_main():
    modelName="ViT"
    version="1"
    os.system("python -u main.py "+modelName+" "+version+" > ../result/specific/"+modelName+"/"+version+".txt")
    modelName="MLPMixer"
    version="1"
    os.system("python -u main.py "+modelName+" "+version+" > ../result/specific/"+modelName+"/"+version+".txt")
    modelName="ConvMixer"
    version="1"
    os.system("python -u main.py "+modelName+" "+version+" > ../result/specific/"+modelName+"/"+version+".txt")
    modelName="ResNet"
    version="1"
    os.system("python -u main.py "+modelName+" "+version+" > ../result/specific/"+modelName+"/"+version+".txt")

if __name__=="__main__":
    
    # create_dataset()
    # draw_accuracy_curve()
    # run_main()
    print("All is well!")
























# images,labels=cutmix(images,labels,num_classes=1020)
        # if (i + 1) % accum_iter == 0 or i + 1 == len(train_loader):
            # optimizer.step(loss)        
        # acc = np.sum(pred == np.argmax(labels.data, axis=1))
        # if i%100==0:
            # print(i,loss.data[0])