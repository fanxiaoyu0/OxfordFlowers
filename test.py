# 测试某一个pkl在test上的精度
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
from tools.train import *
from tools.dataloader import *

model = ConvMixer_768_32()

pkl_path = 'date/.pkl'
model.load('./model/train_result/{}/{}'.format(model.__class__.__name__, pkl_path))
_, _, testdataset = dataloader()
test_acc = valid_one_epoch(model, testdataset, 0)
print(f'test_acc: {test_acc:.3f}')