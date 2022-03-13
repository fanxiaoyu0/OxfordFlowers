from jittor import transform
import jittor as jt
import os

batch_size = 16
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

def dataloader():
    data_dir = './data'
    image_datasets = {x: jt.dataset.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                    ['train', 'valid', 'test']}
    traindataset = image_datasets['train'].set_attrs(batch_size=batch_size, shuffle=True)
    validdataset = image_datasets['valid'].set_attrs(batch_size=64, shuffle=False)
    testdataset = image_datasets['test'].set_attrs(batch_size=1, shuffle=False)

    # Optional return value
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    train_num = len(traindataset)
    val_num = len(validdataset)
    test_num = len(testdataset)
    
    return traindataset, validdataset, testdataset