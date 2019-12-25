import os
import torch
import time
import torch.nn as nn
import tqdm
import numpy as np
import pandas as pd
import pretrainedmodels
from torch.utils.data import DataLoader
from torchvision import transforms, models
from dataset import Resize, AllImageDataset

BATCH_SIZE = 64
num_classes = 40
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_ft = models.resnet18(pretrained=True)
mean = [0.20903548, 0.21178319, 0.21442725]
std = [0.12113936, 0.12205944, 0.12315971]
# print(img_size, mean, std)
data_transforms = transforms.Compose([
    Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

dataset = AllImageDataset('./tot_data', data_transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# print(model_ft)
for name, params in model_ft.named_parameters():
    params.requires_grad = False
model_ft_ftr = model_ft.fc.in_features
model_ft.fc = nn.Linear(model_ft_ftr, num_classes)
model_ft.to(device)

img_paths = np.array([])
pros = np.array([])
preds = np.array([])
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
        print(paths.shape, pro.shape, pred.shape)
        img_paths = np.concatenate((img_paths, paths))
        pros = np.concatenate((pros, pro))
        preds = np.concatenate((preds, pred))
        print(img_paths.shape, pros.shape, preds.shape)
        tot += 1
        if tot >= 2:
            break
img_paths = img_paths.tolist()
pros = pros.tolist()
preds = preds.tolist()
print(len(img_paths), len(pros))
dataframe = pd.DataFrame({'image_path': img_paths, 'image_pro': pros, 'image_pred': preds})
dataframe.to_csv('classify_test.csv', index=False, sep=',')

