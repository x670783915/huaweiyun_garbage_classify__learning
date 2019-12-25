import numpy as np
import pandas as pd
import os

csv_dir = './csvs/ext'
csv_paths = os.listdir(csv_dir)
csv_datas = []
for csv_path in csv_paths:
    csv_data = pd.read_csv(os.path.join(csv_dir, csv_path))
    csv_datas.append(csv_data)
img_paths = np.array(csv_datas[0]['image_path'])

img_classes = np.array(csv_datas[0]['image_pred']).astype(np.int)
img_classes = img_classes.reshape(img_classes.shape[0], 1)
img_pros = np.array(csv_datas[0]['image_pro'])
img_pros = img_pros.reshape(img_pros.shape[0], 1)

for index, csv_data in enumerate(csv_datas):
    if index == 0:
        continue
    img_class = np.array(csv_data['image_pred']).astype(np.int)
    img_class = img_class.reshape(img_class.shape[0], 1)
    img_pro = np.array(csv_data['image_pro'])
    img_pro = img_pro.reshape(img_pro.shape[0], 1)
    img_classes = np.hstack((img_classes, img_class))
    img_pros = np.hstack((img_pros, img_pro))

# print(img_classes.shape)
select_imgs = []
select_img_labels = []
select_img_paths = []
tot = 0
for index, line in enumerate(img_classes):
    # print(line)
    cnt = np.bincount(line)
    # print(cnt)
    max_cnt_num = np.argmax(cnt)
    if cnt[max_cnt_num] >= 4:
        select_imgs.append(index)
        select_img_labels.append(max_cnt_num)
        select_img_paths.append(img_paths[index])
    # tot += 1
    # if tot >= 3:
    #     break

print(len(select_img_paths), len(select_img_labels))
dataframe = pd.DataFrame({'image_path': select_img_paths, 'image_label': select_img_labels})
dataframe.to_csv('ext_data_4.csv', index=False, sep=',')