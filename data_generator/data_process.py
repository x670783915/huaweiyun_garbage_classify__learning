import pandas as pd
import glob
import os

def create_csv():
    img_paths = []
    img_labels = []

    paths = glob.glob('./train_data_v2/' + '*.txt')
    for i, path in enumerate(paths):
        with open(path) as f:
            buf = f.readline()
            img_path, label = buf.split(', ')
            print(i, img_path, label)
            img_paths.append(img_path)
            img_labels.append(label)

    dataframe = pd.DataFrame({'image_path': img_paths, 'image_label': img_labels})
    dataframe.to_csv('test.csv', index=False, sep=',')


def delete_txt():
    paths = glob.glob('./train_data_v2/' + '*.txt')
    for path in paths:
        os.remove(path)

def delete_error_img():
    pass

if __name__ == '__main__':
