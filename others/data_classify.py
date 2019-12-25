import os
import torch
import time
import torch.nn as nn
import tqdm
import numpy as np
import pandas as pd
import pretrainedmodels
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import Resize, AllImageDataset

BATCH_SIZE = 256
num_classes = 40
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_names = [
    'resnext101_32x4d', 
    'resnext101_64x4d', 
    'se_resnext50_32x4d', 
    'se_resnext101_32x4d'
]
pretrained_models = [
    './models/resnext101_32x4d.pth',
    './models/resnext101_64x4d.pth',
    './models/se_resnext50_32x4d.pth',
    './models/se_resnext101_32x4d.pth'
]

for model_name, model_path in zip(model_names, pretrained_models):
    print('{} start eval unclassify dataset'.format(model_name))

    model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=None)
   
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

    dataset = AllImageDataset('./tot_data', data_transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    # print(model_ft)
    for name, params in model_ft.named_parameters():
        params.requires_grad = False
    model_ft_ftr = model_ft.last_linear.in_features
    model_ft.last_linear = nn.Linear(model_ft_ftr, num_classes)
    model_ft.load_state_dict(torch.load(model_path))
    print('load pretrainmodel from {} success'.format(model_path))
    model_ft.to(device)

    img_paths = []
    pros =  []
    preds =  []
    model_ft.eval()
    tot = 0
    with torch.no_grad():
        for inputs, paths in dataloader:
            inputs = inputs.to(device)
            outputs = model_ft(inputs)
            pro, pred = torch.max(outputs, 1)
            pro = pro.cpu().numpy()
            pred = pred.cpu().numpy()
            paths = np.array(paths)
            # print(paths.shape, pro.shape, pred.shape)
            img_paths = np.concatenate((img_paths, paths))
            pros = np.concatenate((pros, pro))
            preds = np.concatenate((preds, pred))
            # print(img_paths.shape, pros.shape, preds.shape)
#             tot += 1
#             if tot >= 3:
#                 break
    img_paths = img_paths.tolist()
    pros = pros.tolist()
    preds = preds.tolist()
    # print(len(img_paths), len(pros))
    dataframe = pd.DataFrame({'image_path': img_paths, 'image_pro': pros, 'image_pred': preds})
    dataframe.to_csv('./csvs/' + model_name + '.csv', index=False, sep=',')
    print(model_name + 'classify success and save file in ' + './csvs/' + model_name + '.csv' )
