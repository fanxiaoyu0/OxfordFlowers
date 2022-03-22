# 测试某一个pkl在test上的精度，并绘制混淆矩阵
from cProfile import label
import jittor as jt
from jittor import nn
from jittor.dataset import Dataset
from jittor.lr_scheduler import CosineAnnealingLR, MultiStepLR
from conv_mixer import ConvMixer
import sys
import argparse
import numpy as np
from PIL import Image           
from tqdm import tqdm       
import datetime    
import json
import matplotlib as plt

# from model import *
# from tools.train import *
# from tools.dataloader import *

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# https://mirrors.aliyun.com/pypi/simple/

def plot_confusion_matrix(confusion_matrix, labels, title):
    #参考https://blog.csdn.net/kane7csdn/article/details/83756583绘制热力的混淆矩阵
    plt.figure(figsize=(20, 20))
    plt.imshow(confusion_matrix, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    num = np.array(range(len(labels)))
    plt.xticks(num, labels, rotation=90)
    plt.yticks(num, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("./ConvMixer.png")

model = ConvMixer(dim = 768, depth = 32, kernel_size=7, patch_size=7,n_classes=102)
jt.flags.use_cuda = 1

pkl_path = '../model/ConvMixer/best.pkl'
model.load(pkl_path)
jt.set_global_seed(648)  

# region Processing data 
resizedImageSize = 256#trial.suggest_int("resizedImageSize", 256, 512,64)
croppedImagesize=resizedImageSize-32
data_transforms = {
    'train': transform.Compose([
        transform.Resize((resizedImageSize,resizedImageSize)),
        transform.RandomCrop((croppedImagesize, croppedImagesize)),       # 从中心开始裁剪
        transform.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
        transform.RandomVerticalFlip(p=0.5),    # 随机垂直翻转
        # transform.RandomRotation(90),
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
batch_size = 16#trial.suggest_int("batch_size", 4, 32)
data_dir = '../data'
image_datasets = {x: jt.dataset.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                ['train', 'valid', 'test']}
traindataset = image_datasets['train'].set_attrs(batch_size=batch_size, shuffle=True)
validdataset = image_datasets['valid'].set_attrs(batch_size=batch_size, shuffle=False)
testdataset = image_datasets['test'].set_attrs(batch_size=1, shuffle=False)
_, _, testdataset = dataloader()


model.eval()

total_acc = 0
total_num = 0
pred_list = list()
true_list = list()

pbar = tqdm(testdataset, desc='Epoch 1 [VALID]')
map_dict = testdataset.class_to_idx
values = list(map_dict.values())
keys = list(map_dict.keys())

for i, (images, labels) in enumerate(pbar):
    output = model(images)
    pred = np.argmax(output.data, axis=1)
    acc = np.sum(pred == labels.data)
    
    total_acc += acc
    total_num += labels.shape[0]

    pred_list.append(int(keys[values.index(int(pred))]))
    true_list.append(int(keys[values.index(int(labels.data))]))

    pbar.set_description(f'Epoch 1 acc={total_acc / total_num:.2f}')

acc = total_acc / total_num
print(f'test_acc: {acc:.3f}')

cm = confusion_matrix(true_list, pred_list)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

label_list = list()
for i in range(0,102):
    if i % 5 == 0:
        label_list.append(f"{i}")
    else:
        label_list.append("")

plot_confusion_matrix(cm, label_list, "ConvMixer")
np.savetxt("./ConvMixer.csv",cm,delimiter=',')