import random
import math
import torch
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms

class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
    
    def __call__(self, img):
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w + w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h + h_padding))
        return img.resize(self.size, self.interpolation)

class RandomRotate(object):
    def __init__(self, degree, p=0.5):
        self.degree = degree
        self.p = p
    
    def __call__(self, img):
        if random.random < self.p:
            rotate_degree = random.uniform(-1 * self.degree, self.degree)
            img = img.rotate(rotate_degree, Image.BILINEAR)
        return img

class RandomGaussianBlur(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return img

def get_train_transform(mean, std, size):
    return transforms.Compose([
        Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

def get_test_transform(mean, std, size):
    return transforms.Compose([
        Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.RandomCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

def get_transform(input_size=224, test_size=224, backbone=None):
    mean, std = [0.4995, 0.4642, 0.4140], [0.2771, 0.2705, 0.2660]
    if backbone is not None and backbone in ['pnasnet5large', 'nasnetamobile']:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    transformations = {}
    transformations['train'] = get_train_transform(mean, std, input_size)
    transformations['test'] = get_test_transform(mean, std, test_size)
    return transformations



# train_mean=tensor([0.4995, 0.4642, 0.4140]), train_std=tensor([0.2771, 0.2705, 0.2660])
# val_mean=tensor([0.4914, 0.4562, 0.4063]), val_std=tensor([0.2777, 0.2706, 0.2656])