import json
from pathlib import Path

import cv2
import h5py
import pandas as pd
import torch
from clearml import InputModel
from tqdm import tqdm

from scripts.seg_grazpedwri_dataset import LightSegGrazPedWriDataset
from unet.classic_u_net import UNet

device = "cuda:3" if torch.cuda.is_available() else "cpu"

model_id = '0427c1de20c140c5bff7284c7a4ae614'
cl_model = InputModel(model_id)
model = UNet.load(cl_model.get_weights(), device).eval().to(device)
H, W = 384, 224

n_files = 500
img_dir = Path('data/img_only_front_all_left')
available_files = pd.read_csv(f'data/{n_files}unlabeled_sample.csv', index_col='filestem').index.tolist()

# create h5 file
h5py_path = Path(f'data/seg_masks/init_{model_id}_{n_files}.h5')
h5py_file = h5py.File(h5py_path, 'w')
# store labels and their index
h5py_file.attrs['labels'] = json.dumps(LightSegGrazPedWriDataset.BONE_LABEL_MAPPING)

for img_name in tqdm(available_files, unit='img', desc='Predict segmentation'):
    img_file = img_dir / (img_name + '.png')
    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    img = torch.from_numpy(img).view(1, 1, H, W).float() / 255
    with torch.inference_mode():
        img = (img - LightSegGrazPedWriDataset.IMG_MEAN) / LightSegGrazPedWriDataset.IMG_STD
        y_hat = model(img.to(device)).squeeze(0)
        y_hat = torch.sigmoid(y_hat) > 0.5
    unet_mask = y_hat.cpu().numpy()

    h5py_file.create_dataset('segmentation_mask/' + img_name, data=unet_mask, compression='gzip',
                             compression_opts=9)
h5py_file.close()
