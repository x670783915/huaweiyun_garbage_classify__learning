import torch
import torch.nn as nn
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
import glob
import pandas as pd
from PIL import Image

from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.utils import make_grid

class GarbageDataset(Dataset):

    def __init__(self, root_dir, csv_path, transforms=None):
        self.root_dir = root_dir
        self.img_paths = []
        self.img_labels = []
        self.transforms = transforms

        csv_data = pd.read_csv(csv_path)
        for path, label in zip(csv_data['image_path'], csv_data['image_label']):
            self.img_paths.append(path)
            self.img_labels.append(label)
        
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.img_paths[index])
        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, self.img_labels[index]

    def __len__(self):
        return len(self.img_labels)

class AllImageDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.img_paths = []
        self.transforms = transforms

        for img_name in os.listdir(root_dir):
            self.img_paths.append(img_name)
        
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.img_paths[index])
        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, img_path

    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':
    # dataset = GarbageDataset('./train_data_v2/', './test.csv')
    dataset = AllImageDataset('./tot_data')
    
class Resize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
    def __call__(self, img):
        radio = self.size[0] / self.size[1]
        w, h = img.size
        if (w*1.0 / h) < radio:
            t = int(h * radio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w + w_padding, h))
        else:
            t = int(w / radio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h + h_padding))
        img = img.resize(self.size, self.interpolation)
        return img