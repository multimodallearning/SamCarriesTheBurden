import random

import h5py
import json
from pathlib import Path
from matplotlib import pyplot as plt
import cv2

img_path = Path('data/img_only_front_all_left')

f = h5py.File('data/seg_masks/sam_0427c1de20c140c5bff7284c7a4ae614_500.h5', 'r')
lbl_idx_mapping = json.loads(f.attrs['labels'])
ds_seg_masks = f['segmentation_mask']

available_files = list(ds_seg_masks.keys())
file_to_show = random.sample(available_files, 1)[0]
img = cv2.imread(str(img_path / (file_to_show + '.png')), cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (224, 384), interpolation=cv2.INTER_NEAREST)
seg_masks = ds_seg_masks[file_to_show][:]
est_dice = ds_seg_masks[file_to_show].attrs['estimated_dice']
for lbl, lbl_idx in lbl_idx_mapping.items():
    if not seg_masks[lbl_idx].any():
        continue

    plt.figure(lbl)
    plt.imshow(img, cmap='gray')
    plt.imshow(seg_masks[lbl_idx], alpha=seg_masks[lbl_idx].astype(float))
    plt.title(lbl + f' (est. dice: {est_dice[lbl_idx]:.4f})')
    plt.show()
