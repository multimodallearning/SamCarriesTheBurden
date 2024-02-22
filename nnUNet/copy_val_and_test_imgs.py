from pathlib import Path

import cv2
from tqdm import tqdm

from scripts.seg_grazpedwri_dataset import LightSegGrazPedWriDataset

img_size = (224, 384)

src = Path('/home/ron/Documents/SemiSAM/data/img_only_front_all_left')
dst = Path('/home/ron/Desktop/img_val_test_all_left')
file2copy = LightSegGrazPedWriDataset('val').available_file_names + LightSegGrazPedWriDataset(
    'test').available_file_names

# copy files
for file in tqdm(file2copy, desc='Copying files'):
    # load image, resize and save
    img = cv2.imread(str(src.joinpath(file).with_suffix('.png')), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    assert cv2.imwrite(str(dst.joinpath(file + '_0000').with_suffix('.png')), img)
