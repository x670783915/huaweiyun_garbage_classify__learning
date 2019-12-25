import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image

def plot_csv(csv_path):
    csv_data = pd.read_csv(csv_path)
    img_classes = np.array(csv_data['image_label']).astype(np.int)

    fig, ax = plt.subplots()
    stat0 = ax.hist(img_classes, 40, histtype='bar')
    ax.set_title('Count')

    fig.tight_layout()
    plt.show()

def plot_csv_pro(csv_path):
    csv_data = pd.read_csv(csv_path)

    img_classes = np.array(csv_data['image_pred']).astype(np.int)
    img_pros = np.array(csv_data['image_pro'])

    fig, ax = plt.subplots(2, 1)
    stat0 = ax[0].hist(img_classes, 40, histtype='bar')
    ax[0].set_title('Count')
    stat1 = ax[1].hist(img_pros, 40, histtype='bar')
    ax[1].set_title('Probabilities')

    fig.tight_layout()
    plt.show()

def plot_dir(dir_path):
    csv_dir = dir_path
    csv_paths = os.listdir(csv_dir)
    csv_datas = []
    for csv_path in csv_paths:
        csv_data = pd.read_csv(os.path.join(csv_dir, csv_path))
        csv_datas.append(csv_data)

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

    fig, ax = plt.subplots(2, 1)
    ax[0].hist(img_classes, 40, histtype='bar')
    ax[0].set_title('Class Cnt')
    ax[1].hist(img_pros, 40, histtype='bar')
    ax[1].set_title('Probabilities')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    csv_path = './others/test.csv'
    plot_csv(csv_path)