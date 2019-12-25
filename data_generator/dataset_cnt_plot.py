import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

csv_paths = ['./train.csv', './test.csv', './val.csv']
csv_datas = []
for csv_path in csv_paths:
    csv_data = pd.read_csv(csv_path)
    csv_datas.append(csv_data)


img_classes = []
for index, csv_data in enumerate(csv_datas):
    img_class = np.array(csv_data['image_label']).astype(np.int)
    img_classes.append(img_class)

fig, ax = plt.subplots()
colors = ['red', 'tan', 'lime']
labels = ['train', 'test', 'val']
ax.hist(img_classes, 40, histtype='bar', color=colors, label=labels)
ax.legend(prop={'size': 10})
fig.tight_layout()
plt.show()
