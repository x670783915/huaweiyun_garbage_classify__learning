import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import time
import os
import pretrainedmodels
import torch.optim as optim
from dataset import GarbageDataset
from utils import Bar, Logger, AverageMeter, accuracy, savefig, get_optimizer, save_checkpoint, save_model
from train_utils import train, test
from nets import se_resnext101_32x4d
from ops import FocalLoss
from transform import get_train_transform
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

best_acc = 0
start_epoch = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_class = 40
batch_size = 256
tot_epoch = 30
lr = 1e-3

# model
pretrainedmodel = 'se_resnext101_32x4d'
pretrainedmodel_path = './pretrain_models/se_resnext101_32x4d-3b2fe3d8.pth'
model, train_layers = se_resnext101_32x4d(40, pretrainedmodel_path)
model.to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
# criterion = FocalLoss(num_class).to(device)
criterion = nn.CrossEntropyLoss().to(device)
# lr_scheduler = None
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=4, verbose=False)

# -------------  Dataset  ---------------
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
train_dataset = GarbageDataset(
    './garbage_dataset_v1/train', 
    './garbage_dataset_v1/train.csv', 
    transforms=get_train_transform(mean, std, 224)
)
val_dataset = GarbageDataset(
    './garbage_dataset_v1/val', 
    './garbage_dataset_v1/val.csv', 
    transforms=get_train_transform(mean, std, 224)
)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
# -------------            ---------------
writer = None # 服务器pytorch版本太低
# load train params -------------
title = pretrainedmodel
# resume = None
checkpoint_path = './saves'
logger = Logger(os.path.join(checkpoint_path, 'log.txt'), title=title)
logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
# if resume:
#     print('==> Resuming from checkpoint')
#     assert os.path.isfile(resume), 'Error no checkpoint file'
#     checkpoint_dir = os.path.dirname(resume)
#     chechpoint = torch.load(resume)
#     best_acc = checkpoint['best_acc']
#     start_epoch = chechpoint['epoch']
#     model.load_state_dict(chechpoint['state_dict'])
#     optimizer.load_state_dict(chechpoint['optimizer'])
#     logger = Logger(os.path.join(checkpoint_dir, 'log.txt'), title=title, resume=True)
# else:
#     logger = Logger(os.path.join(checkpoint_path, 'log.txt'), title=title)
#     logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


# train ------------
for epoch in range(start_epoch, tot_epoch):
    print('\nEpoch [%d | %d] LR: %f' % (epoch + 1, tot_epoch, optimizer.param_groups[0]['lr']))
    
    train_loss, train_acc, train_5 = train(train_dataloader, model, criterion, optimizer, epoch, device, writer)
    test_loss, test_acc, test_5 = test(val_dataloader, model, criterion, epoch, device, writer)

    # logger.append([ lr, train_loss, test_loss, train_acc, test_acc ])
    print('train_loss:%f, val_loss:%f, train_acc:%f, train_5:%f, val_acc:%f, val_5:%f' % (train_loss, test_loss, train_acc, train_5, test_acc, test_5))

    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)

    save_checkpoint({
        'fold': 1,
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'train_acc': train_acc,
        'acc': test_acc,
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
    }, is_best, single=False, checkpoint=checkpoint_path, filename='test')

    if lr_scheduler is not None:
        lr_scheduler.step(test_loss)

logger.close()
logger.plot()
savefig(os.path.join(checkpoint_path, 'log.eps'))
print('Best Acc:{}'.format(best_acc))
