import os
import glob
import io
from tqdm import tqdm
from PIL import Image

download_dir = './downloads'
img_dirs = ['./fuse_dataset']

for dir_name in os.listdir(download_dir):
    img_dirs.append(os.path.join(download_dir, dir_name))

img_paths = []
for img_dir in img_dirs:
    img_names = glob.glob(img_dir + '/*.*')
    for img_name in img_names:
        img_paths.append(img_name)


img_save_dir = './dataset/garbage/ext_data/'
error_num = 0
total_img = len(img_paths)
img_name_index = 0
for img_path in tqdm(img_paths):
    if img_path.endswith('.txt'):
        total_img -= 1
        continue
    is_valid = True
    try:
        with open(img_path, 'rb') as img_file:
            img_byte = img_file.read()
        img_file = io.BytesIO(img_byte)
        img = Image.open(img_file).convert('RGB')
        img.save(img_save_dir + 'img_v3_' + str(img_name_index) + '.jpg')
        img_file.close()
        img.close()
    except Exception as e:
        is_valid = False
        print(e)

    if is_valid:
        img_name_index += 1
    else:
        print(img_path)
        error_num += 1
        if error_num >= 100:
            break

print(img_name_index)
print(error_num)
print(total_img)
