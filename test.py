import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import time
import os
import pretrainedmodels
from dataset import GarbageDataset
from train_utils import train, test, test_a
from nets import se_resnext101_32x4d
from ops import FocalLoss
from transform import get_test_transform
from torch.utils.data import DataLoader

best_acc = 0
start_epoch = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_class = 40
batch_size = 256
tot_epoch = 30
lr = 1e-3

# model
pretrainedmodel = 'se_resnext101_32x4d'
pretrainedmodel_path = './pretrain_models/...' # 具体设置
model, _ = se_resnext101_32x4d(40)
model.to(device)
# criterion = FocalLoss(num_class).to(device)
criterion = nn.CrossEntropyLoss().to(device)

classes = (
    '其他垃圾/一次性快餐盒', '其他垃圾/污损塑料', '其他垃圾/烟蒂', '其他垃圾/牙签',
    '其他垃圾/破碎花盆及碟碗', '其他垃圾/竹筷',
    '厨余垃圾/剩饭剩菜', '厨余垃圾/大骨头', '厨余垃圾/水果果皮', '厨余垃圾/水果果肉',
    '厨余垃圾/茶叶渣', '厨余垃圾/菜叶菜根','厨余垃圾/蛋壳','厨余垃圾/鱼骨',
    '可回收物/充电宝','可回收物/包','可回收物/化妆品瓶','可回收物/塑料玩具',
    '可回收物/塑料碗盆','可回收物/塑料衣架',
    '可回收物/快递纸袋','可回收物/插头电线','可回收物/旧衣服',
    '可回收物/易拉罐','可回收物/枕头','可回收物/毛绒玩具','可回收物/洗发水瓶',
    '可回收物/玻璃杯','可回收物/皮鞋','可回收物/砧板','可回收物/纸板箱',
    '可回收物/调料瓶','可回收物/酒瓶','可回收物/金属食品罐','可回收物/锅',
    '可回收物/食用油桶','可回收物/饮料瓶',
    '有害垃圾/干电池','有害垃圾/软膏','有害垃圾/过期药物'
)
mean, std = [0.4995, 0.4642, 0.4140], [0.2771, 0.2705, 0.2660]
test_dataset = GarbageDataset(
    './garbage_dataset_v1/test', 
    './garbage_dataset_v1/test.csv', 
    transforms=get_test_transform(mean, std, 224)
)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)


class_probs, class_preds, test_acc, test_5 = test_a(test_dataloader, model, criterion, 0, device, None)
print(class_probs, class_preds, test_acc, test_5)
state = {}
state['class_probs'] = class_probs
state['class_preds'] = class_preds
state['test_acc'] = test_acc
state['test_5'] = test_5
torch.save(state. 'test_out.pth')
