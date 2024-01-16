import json

import cv2
import h5py
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from prompt_utils import PromptExtractor

img_name = '0001_1297860395_01_WRI-L1_M014'
image = cv2.imread(f'/home/ron/Documents/SemiSAM/notebooks/images/{img_name}.png', cv2.IMREAD_GRAYSCALE)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
image = cv2.resize(image, (224, 384), interpolation=cv2.INTER_LINEAR)

f = h5py.File('/home/ron/Documents/KidsBoneChecker/datasets/data/GRAZPEDWRI-DX/SegGraz_nnunet_predictions.h5', 'r')
lbl_idx_mapping = json.loads(f.attrs['labels'])
ds_seg_masks = f['nnUNet_prediction']

seg_masks = torch.from_numpy(ds_seg_masks[img_name][:])
prompt_extractor = PromptExtractor(seg_masks)
prompts = prompt_extractor.extract()

for prompt in prompts:
    lbl = list(lbl_idx_mapping.keys())[prompt.class_idx]
    plt.figure(lbl)
    plt.imshow(image)
    # seeds
    plt.scatter(prompt.pos_seeds[:, 0], prompt.pos_seeds[:, 1], c='r')
    plt.scatter(prompt.neg_seeds[:, 0], prompt.neg_seeds[:, 1], c='b')

    # box
    if prompt.box is not None:
        box = prompt.box
        x_min, y_min, x_max, y_max = box
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

    plt.title(lbl)
    plt.show()
