import json
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from scripts.seg_grazpedwri_dataset import LightSegGrazPedWriDataset


@torch.no_grad()
def bin2dec(x: torch.Tensor):
    assert x.ndim == 4, 'Input tensor should be format BxCxHxW'
    num_bits = x.shape[1]
    mask = 2 ** torch.arange(num_bits, dtype=torch.int, device=x.device).view(1, -1, 1, 1)
    x_decimal = x * mask
    return x_decimal.sum(dim=1, keepdim=True).int()


@torch.no_grad()
def dec2bin(x: torch.Tensor, num_bits: int):
    assert x.ndim == 4, 'Input tensor should be format BxCxHxW'
    assert (x <= 2 ** num_bits).all(), f'To less bits for proper representation'
    mask = 2 ** torch.arange(num_bits, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    x_binary = x.bitwise_and(mask).bool()
    return x_binary


H, W = 384, 224

save_base_dir = Path('/home/ron/Downloads/nnUNet_datasets')
save_base_dir.mkdir(exist_ok=True)

for train_samples in tqdm([1, 5, 10, 15, 20, 25, 30, 35, 'all']):
    ds = LightSegGrazPedWriDataset('train', number_training_samples=train_samples)
    if train_samples == 'all':
        train_samples = len(ds)

    # create folder structure
    dataset_name = f'Dataset{100 + train_samples}_Light_Seg_Graz_bce_all_left'
    save_dir = save_base_dir.joinpath(dataset_name)
    save_dir.mkdir(exist_ok=True, parents=True)
    save_dir.joinpath('labelsTr').mkdir(exist_ok=True, parents=True)
    save_dir.joinpath('imagesTr').mkdir(exist_ok=True, parents=True)

    lbl_tmp = torch.cat([bin2dec(y.unsqueeze(0)) for _, y, _ in ds], dim=0)
    lbl_idx_mapping = {'background': 0}
    for i, lbl_comb in enumerate(lbl_tmp.unique().int()):
        if lbl_comb.item() == 0:  # skip background
            continue
        lbl_idx_mapping[lbl_comb.item()] = i
    del lbl_tmp

    for img, lbl_bin, file_name in ds:
        # generate label image
        lbl_dec = bin2dec(lbl_bin.unsqueeze(0)).squeeze()
        lbl = torch.zeros_like(lbl_dec)
        for bce_int_code, lbl_idx in lbl_idx_mapping.items():
            if bce_int_code == 'background':  # identity mapping
                continue

            lbl[lbl_dec == bce_int_code] = lbl_idx
        lbl = lbl.byte().numpy()
        assert cv2.imwrite(str(save_dir.joinpath('labelsTr').joinpath(file_name).with_suffix('.png')), lbl)

        # save img
        img = img * 255
        img = img.squeeze(0).byte().numpy()
        assert cv2.imwrite(str(save_dir.joinpath('imagesTr').joinpath(file_name + '_0000').with_suffix('.png')), img)

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
