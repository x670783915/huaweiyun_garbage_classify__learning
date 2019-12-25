import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image

def mkdir(path):  # 判断是否存在指定文件夹，不存在则创建
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        return True
    else:

        return False


class FeatureVisualization():
    
    def __init__(self, img_path, selected_layer):
        self.img_path = img_path
        self.selected_layer = selected_layer
        self.pretrain_model = models.vgg16(pretrained=True).features
        # mean and std list for channels (Imagenet)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def precess_image(self):
        img = Image.open(self.img_path)
        img = self.transform(img)
        return Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    
    def get_feature(self):
        img = self.precess_image()
        print('input shape: {}'.format(img.shape))
        x = img
        for index, layer in enumerate(self.pretrain_model):
            x = layer(x)
            if index == selected_layer:
                return x
    
    def get_single_feature(self):
        features = self.get_feature()
        print('features shape: {}'.format(feature.shappe))
        feature = features[:, 0, :, :]
        print('feature shape: {}'.format(feature.shappe))
        feature = feature.view(feature.shape[1], feature.shape[2])
        print('feature shape: {}'.format(feature.shappe))
        return feature
    
    def save_feature_to_img(self):
        features = self.get_single_feature()
        for i in range(features.shape[1]):
            feature = features[:, i, :, :]
            feature = feature.view(feature.shape[1], feature.shape[2])
            feature = feature.data.numpy()
            feature = 1.0 / (1 + np.exp(-1 * feature))
            feature = np.round(feature * 255)
            print(feature[0])
            mkdir('./feature/' + str(self.selected_layer))
            img = Image.fromarray(feature)
            img.save('./feature/' + str( self.selected_layer) + '/' + str(i) + '.jpg')

