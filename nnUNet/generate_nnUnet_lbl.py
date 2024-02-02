import json
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from scripts.seg_grazpedwri_dataset import LightSegGrazPedWriDataset

H, W = 384, 224
# calculated with calculate_ds_split_4_non_overlapping_lbls.py
disjuncted_splits = [
    [0, 2, 9, 10, 14],
    [1, 4, 5, 11, 13],
    [3, 6, 12, 16],
    [3, 8, 12, 15],
    [4, 5, 7, 13, 16],
]

save_base_dir = Path('/home/ron/Downloads/nnUNet_datasets')
save_base_dir.mkdir(exist_ok=True)
img_save_dir = save_base_dir.joinpath('imagesTr')
img_save_dir.mkdir(exist_ok=True)

ds = LightSegGrazPedWriDataset('train')
for split_idx, lbl_split in enumerate(disjuncted_splits):
    lbl_ind_set = [ds.BONE_LABEL[i] for i in lbl_split]
    lbl_idx_mapping = {k: v + 1 for v, k in enumerate(lbl_ind_set)}  # idx for the nnUNet labels (0 is background)
    lbl_idx_mapping['background'] = 0

    # create folder structure
    dataset_name = f'Dataset{40 + split_idx:03d}_Light_Seg_Graz_split{split_idx}_mutex_all_left'
    save_dir = save_base_dir.joinpath(dataset_name)
    save_dir.mkdir(exist_ok=True, parents=True)
    save_dir.joinpath('labelsTr').mkdir(exist_ok=True, parents=True)

    for img, lbl, file_name in tqdm(ds, unit='img', desc=f'Processing split {split_idx + 1}/{len(disjuncted_splits)}'):
        selected_lbl = lbl[lbl_split].bool()
        # add background
        background = torch.logical_not(selected_lbl.any(dim=0, keepdim=True))
        selected_lbl = torch.cat([background, selected_lbl], dim=0)

        # generate label image
        assert selected_lbl.sum(0).max() == 1, f'Overlapping labels in {file_name}'
        lbl = selected_lbl.int().argmax(0).byte().numpy()
        assert cv2.imwrite(str(save_dir.joinpath('labelsTr').joinpath(file_name).with_suffix('.png')), lbl)

        # save img
        img = img * 255
        img = img.squeeze(0).byte().numpy()
        assert cv2.imwrite(str(img_save_dir.joinpath(file_name + '_0000').with_suffix('.png')), img)

    # create dataset.json
    dataset_json = {
        "channel_names": {
            "0": "zscore"
        },
        "labels": lbl_idx_mapping,
        "numTraining": len(ds),
        "file_ending": ".png"
    }

    with open(str(save_dir.joinpath('dataset.json')), 'w') as f:
        json.dump(dataset_json, f, indent=4)
