import os
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

jt.flags.use_cuda = 1

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, accum_iter, scheduler):
    model.train()
    total_acc = 0
    total_num = 0
    losses = []

    # pbar = tqdm(train_loader, desc=f'Epoch {epoch} [TRAIN]')
    for i, (images, labels) in enumerate(train_loader):
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

        
        # pbar.set_description(f'Epoch {epoch} loss={sum(losses) / len(losses):.2f} '
                            #  f'acc={total_acc / total_num:.2f}')
    scheduler.step()
    return round(total_acc/total_num,3)
    # run["train/loss"].log(round(sum(losses) / len(losses),2))
    # run["train/acc"].log(round(total_acc / total_num,2))

def valid_one_epoch(model, val_loader, epoch):
    model.eval()
    total_acc = 0
    total_num = 0

    # pbar = tqdm(val_loader, desc=f'Epoch {epoch} [VALID]')
    for i, (images, labels) in enumerate(val_loader):
        output = model(images)
        pred = np.argmax(output.data, axis=1)

        acc = np.sum(pred == labels.data)
        total_acc += acc
        total_num += labels.shape[0]

        # pbar.set_description(f'Epoch {epoch} acc={total_acc / total_num:.2f}')

    acc = total_acc / total_num
    # run["eval/acc"].log(round(acc,2))
    return round(acc,3)

def trial(learningRate=1e-5):
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
    # testdataset = image_datasets['test'].set_attrs(batch_size=1, shuffle=False)
    # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    # train_num = len(traindataset)
    # val_num = len(validdataset)
    # test_num = len(testdataset)
    # endregion

    # region model and optimizer
    depth=12#trial.suggest_int("depth", 4, 16)
    dropout=0#trial.suggest_float("dropout", 0, 0.9)
    dim=512
    patch_size=16#trial.suggest_int("batch_size", 4, 16)
    # model=MLPMixerForImageClassification(
    #     in_channels=3,patch_size = patch_size, d_model = dim, depth = depth, num_classes=102,\
    #     image_size=croppedImagesize,dropout=dropout)
    # model=ConvMixer(dim = 768, depth = 32, kernel_size=7, patch_size=7,n_classes=102)
    # model = VisionTransformer(patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3.)
    model=Resnet50(num_classes=102)
    criterion = nn.CrossEntropyLoss()
    #lr=2e-5#trial.suggest_float("learningRate", 1e-6,5e-4,log=True)
    # weight_decay=trial.suggest_float("weight_decay", 5e-2, 1e-4)
    optimizer = nn.Adam(model.parameters(), lr=learningRate, weight_decay=1e-4)
    # scheduler = MultiStepLR(optimizer, milestones=[40, 80, 160, 240], gamma=0.2) #learning rate decay
    T_max=15
    scheduler = CosineAnnealingLR(optimizer, 15, 1e-8)
    # endregion

    # region train and valid
    # print("resizedImageSize:",resizedImageSize)
    # print("depth:",depth)
    # print("dropout:",dropout)
    print("----------------- A new trial ---------------------")
    print("learning rate:",learningRate)
    epochs = 800
    currentBestAccuracy=0.0
    currentBestEpoch=0
    for epoch in range(epochs):
        trainAccuracy=train_one_epoch(model, traindataset, criterion, optimizer, epoch, 1, scheduler)
        validAccuracy=valid_one_epoch(model, validdataset, epoch)
        print("epoch:",epoch,"validAccuracy:",validAccuracy,"trainAccuracy:",trainAccuracy,"delta:",round(validAccuracy-currentBestAccuracy,3))
        if validAccuracy > currentBestAccuracy:
            currentBestAccuracy=validAccuracy
            currentBestEpoch=epoch
        # Report intermediate objective value.
        # trial.report(best_acc, epoch)
        # Handle pruning based on the intermediate value.
        # if trial.should_prune():
        #     raise optuna.TrialPruned()
    return currentBestAccuracy,currentBestEpoch

if __name__=="__main__":
    globalBestAccuracy=0.0
    learningRateList=[1e-4]#[1e-6,1e-5,5e-5,1e-4,1e-3]
    trialBoard=[]
    for learningRate in learningRateList:
        accuracy,epoch=trial(learningRate=learningRate)
        trialBoard.append("learningRate: "+str(learningRate)+" accuracy:"+str(accuracy)+" epoch: "+str(epoch))
        print("=============================================================================================")
        print("learningRate: "+str(learningRate)+" accuracy:"+str(accuracy)+" epoch: "+str(epoch)+" delta: "+str(round(accuracy-globalBestAccuracy,3)))
        print("=============================================================================================")
        if accuracy>globalBestAccuracy:
            globalBestAccuracy=accuracy
    for  oneTrial in trialBoard:
        print(oneTrial) 
    # Add stream handler of stdout to show the messages
    # optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    # study = optuna.create_study(direction="maximize",pruner=optuna.pruners.MedianPruner(n_warmup_steps=30))
    # study.optimize(objective, n_trials=1, callbacks=[neptune_callback])
    # best_params = study.best_params
    # print(best_params)
    # run.stop()

