import os
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import StratifiedKFold
from PIL import Image

csv_path = './dataset_v1.csv'
csv_data = pd.read_csv(csv_path)
img_classes = np.array(csv_data['image_label']).astype(np.int)
img_paths = np.array(csv_data['image_path'])

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
tot = 0
for trn_idx, test_idx in folds.split(img_paths, img_classes):
    train_paths = np.array(img_paths[trn_idx])
    test_paths = list(img_paths[test_idx])
    train_labels = np.array(img_classes[trn_idx])
    test_labels = list(img_classes[test_idx])

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
for trn_idx, val_idx in folds.split(train_paths, train_labels):
    val_paths = list(train_paths[val_idx])
    train_paths = list(train_paths[trn_idx])
    val_labels = list(train_labels[val_idx])
    train_labels = list(train_labels[trn_idx])
    break
    


data_dir = './dataset/garbage/dataset_v1/'
train_dir = './dataset/garbage/train/'
test_dir = './dataset/garbage/test/'
val_dir = './dataset/garbage/val/'

train_data_frame = pd.DataFrame({'image_path': train_paths, 'image_label': train_labels})
train_data_frame.to_csv('./dataset/garbage/train.csv', index=False, sep=',')
test_data_frame = pd.DataFrame({'image_path': test_paths, 'image_label': test_labels})
test_data_frame.to_csv('./dataset/garbage/test.csv', index=False, sep=',')
val_data_frame = pd.DataFrame({'image_path': val_paths, 'image_label': val_labels})
val_data_frame.to_csv('./dataset/garbage/val.csv', index=False, sep=',')

train_len, test_len, val_len = len(train_paths), len(test_paths), len(val_paths)
print(train_len, test_len)
for index, name in enumerate(train_paths):
    img = Image.open(data_dir + name).convert('RGB')
    img.save(train_dir + name)
    print('{}/{} for train data'.format(index, train_len))
for index, name in enumerate(val_paths):
    img = Image.open(data_dir + name).convert('RGB')
    img.save(val_dir + name)
    print('{}/{} for val data'.format(index, val_len))
for index, name in enumerate(test_paths):
    img = Image.open(data_dir + name).convert('RGB')
    img.save(test_dir + name)
    print('{}/{} for test data'.format(index, test_len))


