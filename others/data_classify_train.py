import os
import torch
import time
import torch.nn as nn
import tqdm
import pretrainedmodels
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms, models
from dataset import Resize, GarbageDataset
from train import train_model

model_names = [
    'resnext101_32x4d', 
    'resnext101_64x4d', 
    'se_resnext50_32x4d', 
    'se_resnext101_32x4d'
]

pretrain_model_paths = [
    './pretrain_models/resnext101_32x4d-29e315fa.pth',
    './pretrain_models/resnext101_64x4d-e77a0586.pth',
    './pretrain_models/se_resnext50_32x4d-a260b3a4.pth',
    './pretrain_models/se_resnext101_32x4d-3b2fe3d8.pth'
]

num_classes = 40
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

for index, model_name in enumerate(model_names):

    # prepare data --------------------------------
    model_configs = pretrainedmodels.pretrained_settings[model_name]['imagenet']
    input_size = model_configs['input_size']
    mean = model_configs['mean']
    std = model_configs['std']
    img_size = (input_size[1], input_size[2])
    # print(img_size, mean, std)
    data_transforms = transforms.Compose([
        Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    dataset = GarbageDataset(
                './garbage_classify_v2/train_data_v2/', 
                './test.csv', 
                data_transforms)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    # fine-tune model define --------------------------------
    model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=None)
    model_ft.load_state_dict(torch.load(pretrain_model_paths[index]))
    print('load pretrain model from {}'.format(pretrain_model_paths[index]))
    # print(model_ft)
    for name, params in model_ft.named_parameters():
        params.requires_grad = False

    model_ft_ftr = model_ft.last_linear.in_features
    model_ft.last_linear = nn.Linear(model_ft_ftr, num_classes)

    # for name, params in model_ft.named_parameters():
    #     if params.requires_grad:
    #         print('-' * 20, '\n{} have {} to learn\n'.format(name, params.size()), '-' * 20)

    # learing utils --------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([{'params': model_ft.last_linear.parameters()}], lr=LEARNING_RATE)
    lr_scheduler = None # fine-tune lr_scheduler isnt important

    # train --------------------------------
    train_model(model_ft, dataloader, criterion, optimizer, epochs=20, save_name=model_name)
