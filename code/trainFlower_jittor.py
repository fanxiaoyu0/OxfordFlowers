import os
import jittor as jt
from jittor import nn
from jittor import transform
from jittor.dataset import Dataset
from jittor.lr_scheduler import CosineAnnealingLR, MultiStepLR

import sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from mlp_mixer import MLPMixer_S_16
# import neptune.new as neptune
from conv_mixer import ConvMixer_768_32

# run = neptune.init(
#     project="fanxiaoyu1234/OxfordFlowers",
#     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0\
#     cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhNTEyOTJhYi0zN2E2LTQzMWQtODI3ZC1iNWFjY2M2NDdjMmUifQ==",
# )  # your credentials


jt.flags.use_cuda = 1

# ============== ./tools/util.py ================== # 

# ============== ./models/MoCo(MAE,MaskFeat,....).py ================== # 

# ========== ./tools/pretrain.py ================= # 

# ========== ./tools/train.py ================= # 

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, accum_iter, scheduler):
    model.train()
    total_acc = 0
    total_num = 0
    losses = []

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [TRAIN]')
    for i, (images, labels) in enumerate(pbar):
        # print(images.shape)
        output = model(images)
        loss = criterion(output, labels)

        optimizer.backward(loss)
        if (i + 1) % accum_iter == 0 or i + 1 == len(train_loader):
            optimizer.step(loss)

        # print(output)
        pred = np.argmax(output.data, axis=1)
        acc = np.sum(pred == labels.data)
        total_acc += acc
        total_num += labels.shape[0]
        losses.append(loss.data[0])

        
        pbar.set_description(f'Epoch {epoch} loss={sum(losses) / len(losses):.2f} '
                             f'acc={total_acc / total_num:.2f}')
    scheduler.step()
    # run["train/loss"].log(round(sum(losses) / len(losses),2))
    # run["train/acc"].log(round(total_acc / total_num,2))

def valid_one_epoch(model, val_loader, epoch):
    model.eval()
    total_acc = 0
    total_num = 0

    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [VALID]')
    for i, (images, labels) in enumerate(pbar):
        output = model(images)
        pred = np.argmax(output.data, axis=1)

        acc = np.sum(pred == labels.data)
        total_acc += acc
        total_num += labels.shape[0]

        pbar.set_description(f'Epoch {epoch} acc={total_acc / total_num:.2f}')

    acc = total_acc / total_num
    # run["eval/acc"].log(round(acc,2))
    return acc

# ========== ./datasets/dataloader.py =============== # 

data_transforms = {
    'train': transform.Compose([
        transform.Resize((256,256)),
        transform.RandomCrop((224, 224)),       # 从中心开始裁剪
        transform.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
        transform.RandomVerticalFlip(p=0.5),    # 随机垂直翻转
        transform.ToTensor(),
        transform.ImageNormalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # 均值，标准差
    ]),
    'valid': transform.Compose([
        transform.Resize((224,224)),
        transform.ToTensor(),
        transform.ImageNormalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
    ]),
    'test': transform.Compose([
        transform.Resize((224,224)),
        transform.ToTensor(),
        transform.ImageNormalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
    ])
}

batch_size = 16
data_dir = '../data'
image_datasets = {x: jt.dataset.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                ['train', 'valid', 'test']}
traindataset = image_datasets['train'].set_attrs(batch_size=batch_size, shuffle=True)
validdataset = image_datasets['valid'].set_attrs(batch_size=64, shuffle=False)
testdataset = image_datasets['test'].set_attrs(batch_size=1, shuffle=False)

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
train_num = len(traindataset)
val_num = len(validdataset)
test_num = len(testdataset)

# ========== ./run.py =============== # 

jt.set_global_seed(648)

model=MLPMixer_S_16(num_classes=102)
# model = ConvMixer_768_32(num_classes=102)

# run["parameters"] = {"model":"MLPMixer_S_16","learning_rate": 0.003, "weight_decay":1e-4,"optimizer": "Adam",
#                         "scheduler":"CosineAnnealingLR(optimizer, 15, 1e-5)"}

criterion = nn.CrossEntropyLoss()
optimizer = nn.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
# scheduler = MultiStepLR(optimizer, milestones=[40, 80, 160, 240], gamma=0.2) #learning rate decay
scheduler = CosineAnnealingLR(optimizer, 15, 1e-5)


epochs = 300
best_acc = 0.0
best_epoch = 0
for epoch in range(epochs):
    train_one_epoch(model, traindataset, criterion, optimizer, epoch, 1, scheduler)
    acc = valid_one_epoch(model, validdataset, epoch)
    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
        model.save(f'../model/MLPMixer_S_16/2/MLPMixer_S_16-bestmodel.pkl')
        # model.save(f'ConvMixer-{epoch}-{acc:.2f}.pkl')

print(best_acc, best_epoch)

# run.stop()