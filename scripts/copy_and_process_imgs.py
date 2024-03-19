# script to copy and process (flip to left if necessary) images from the GrazPedWri dataset

from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm
import shutil

src_path = Path('/home/ron/Documents/KidsBoneChecker/datasets/data/GRAZPEDWRI-DX/img8bit')
dst_path = Path('data/img_only_front_all_left')

df_meta = pd.read_csv('data/dataset.csv', index_col='filestem')

# filter out images with wrong projection
available_files = df_meta.index[df_meta['projection'] == 1].tolist()

for file_name in tqdm(available_files, unit='img'):
    assert src_path.joinpath(file_name).with_suffix(
        '.png').exists(), f'Image {file_name} not found in GrazPedWri dataset'

    # flip image of right hand
    need_to_flip = df_meta.loc[file_name, 'laterality'] == 'R'

    # process image
    if need_to_flip:
        img = cv2.imread(str(src_path.joinpath(file_name).with_suffix('.png')), cv2.IMREAD_GRAYSCALE)
        img = cv2.flip(img, 1)
        assert cv2.imwrite(str(dst_path.joinpath(file_name).with_suffix('.png')),
                           img), f'Failed to write image {file_name}'
    else:
        shutil.copy(src_path.joinpath(file_name).with_suffix('.png'), dst_path.joinpath(file_name).with_suffix('.png'))
