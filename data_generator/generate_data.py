import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

train_data_dir = './train_data_v2'
train_data_csv = './train_data.csv'

ext_data_dir = './dataset/garbage/ext_data'
ext_data_csv = './ext_data.csv'

train_csv_data = pd.read_csv(train_data_csv)
ext_csv_data = pd.read_csv(ext_data_csv)

img_paths, labels = [], []
tot = 0
for path, label in zip(train_csv_data['image_path'], train_csv_data['image_label']):
    img_path = train_data_dir + '/' + path
    img_paths.append(img_path)
    # if tot <= 10:
    #     print(img_path)
    #     tot += 1
    labels.append(label)

for path, label in zip(ext_csv_data['image_path'], ext_csv_data['image_label']):
    img_paths.append(path)
    # if tot <= 20:
    #     print(path)
    #     tot += 1
    labels.append(label)

output_dir = './dataset/garbage/dataset_v1/'

error_num = 0
img_name_index = 0
image_names = []
image_label = []
process_len = len(img_paths)
for path, label in zip(img_paths, labels):
    is_valid = True
    img_name = 'img_' + str(img_name_index) + '_' + str(label) + '.jpg'
    try:
        img = Image.open(path)
        img.save(output_dir + img_name)
        img.close()

    except Exception as e:
        is_valid = False
        print(e)
    if is_valid:
        img_name_index += 1
        image_names.append(img_name)
        image_label.append(label)
    else:
        error_num += 1
    print('{} / {}'.format(error_num + img_name_index, process_len))

print(len(image_names), len(image_label))
dataframe = pd.DataFrame({'image_path': image_names, 'image_label': image_label})
dataframe.to_csv('dataset_v1.csv', index=False, sep=',')
