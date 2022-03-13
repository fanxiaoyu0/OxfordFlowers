import jittor as jt
from jittor import nn
from jittor.dataset import Dataset
from jittor.lr_scheduler import CosineAnnealingLR, MultiStepLR

import sys
import argparse
import numpy as np
from PIL import Image           
from tqdm import tqdm       
import datetime    
import json
import matplotlib as plt

from model import *
from tools import *

# 此处可以存储所有想保存的参数
# 加载模型参数的参数为'时间/xx.pkl'
param_dict = {
    'load_flag' : 0,
    'load_pkl'  : '',
    'epochs' : 2
}

if __name__ == '__main__':
    jt.flags.use_cuda = 1
    jt.set_global_seed(648)
    now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    
    model = ConvMixer_768_32()
    if param_dict['load_flag'] == 1:
        if os.path.exists('./model/train_result/{}/{}'.format(model.__class__.__name__, param_dict['load_pkl'])) == True:
            print('load pkl {}/{}'.format(model.__class__.__name__, param_dict['load_pkl']))
            model.load('./model/train_result/{}/{}'.format(model.__class__.__name__, param_dict['load_pkl']))
        else:
            raise Exception('Pkl doesn\'t exist.')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = nn.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[40, 80, 160, 240], gamma=0.2) #learning rate decay
    # scheduler = CosineAnnealingLR(optimizer, 15, 1e-5)

    traindataset, validdataset, _ = dataloader()
    
    epochs = param_dict['epochs']
    best_acc = 0.0
    best_epoch = 0
    
    train_acces, val_acces = [], []
    for epoch in range(epochs):
        train_acc = train_one_epoch(model, traindataset, criterion, optimizer, epoch, 1, scheduler)
        train_acces.append(train_acc)
        val_acc = valid_one_epoch(model, validdataset, epoch)
        val_acces.append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            model.save(f'./model/train_result/{model.__class__.__name__}/{now}/{epoch}-{val_acc:.2f}.pkl')

    print(f'best_acc: {best_acc}, best_epoch: {best_epoch}')
    
    # 存储本次训练的各项参数数据
    jsOb = json.dumps(param_dict, indent=4)
    fileObject = open(f'./model/train_result/{model.__class__.__name__}/{now}/param.json','w')
    fileObject.write(jsOb)
    fileObject.close()
    
    # 绘制本次训练集精度-epoch，验证集精度-epoch图像并保存
    x = np.linspace(1, epoch, epoch)
    plt.plot(x, train_acces, label='train_acc')
    plt.plot(x, val_acces, label='val_acc')
    plt.xlabel('Epoch')
    plt.ylim(0, 1)
    plt.legend()
    plt.save(f'./model/train_result/{model.__class__.__name__}/{now}/train_val.png')