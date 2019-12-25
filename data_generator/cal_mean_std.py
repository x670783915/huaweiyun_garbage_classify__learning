# coding:utf-8
import os
import numpy as np
import torchvision.transforms as transforms
import pickle
from dataset import Resize
from PIL import Image

def generate_mean_std():
    means = [0.0, 0.0, 0.0]
    stdevs = [0.0, 0.0, 0.0]

    img_size = 288

    data_transform = transforms.Compose([
        Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    img_path = './train_data_v2/'
    img_paths = os.listdir(img_path)
    resizeor = Resize((img_size, img_size))
    data_len = len(img_paths)
    for i, name in enumerate(img_paths):
        img = Image.open(img_path + name).convert('RGB')
        img = resizeor(img)
        img = np.array(img)
        for j in range(3):
            means[j] += img[j, :, :].mean()
            stdevs[j] += img[j, :, :].std()
        print('{} / {} was complete '.format(i, data_len))

    means = np.asarray(means) / (data_len * 255)
    stdevs = np.asarray(stdevs) / (data_len * 255)

    print('normMean = {}'.format(means))
    print('normStd = {}'.format(stdevs))

    with open('./mean_and_std.txt', 'wb') as f:
        pickle.dump(means, f)
        pickle.dump(stdevs, f)
    
# [0.20903548 0.21178319 0.21442725]
# [0.12113936 0.12205944 0.12315971]
