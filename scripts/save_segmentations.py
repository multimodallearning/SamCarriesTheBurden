# script to offline generate segmentation masks

import json
from pathlib import Path

import cv2
import h5py
import torch
from clearml import InputModel
from tqdm import tqdm

from custom_arcitecture.classic_u_net import UNet
from scripts.dental_dataset import DentalDataset

device = "cuda:3" if torch.cuda.is_available() else "cpu"

model_id = 'fff060f575994796936422b8c2819c5e'
print(f'Using model: {model_id}')
cl_model = InputModel(model_id)
model = UNet.load(cl_model.get_weights(), device).eval().to(device)
H, W = 224, 384

img_dir = Path('data/DentalSeg/img')
available_files = DentalDataset('train').available_files

# create h5 file
h5py_path = Path(f'data/seg_masks/{model_id}')
h5py_path.mkdir(parents=True, exist_ok=True)
h5py_path = h5py_path / f'raw_segmentations_{len(available_files)}.h5'
h5py_file = h5py.File(h5py_path, 'w')
# store labels and their index
h5py_file.attrs['labels'] = json.dumps({lbl:idx for idx, lbl in enumerate(DentalDataset.CLASS_LABEL)})
h5py_file.attrs['clearml_model_id'] = model_id

for img_name in tqdm(available_files, unit='img', desc='Predict segmentation'):
    img_file = img_dir / (img_name + '.jpg')
    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    img = torch.from_numpy(img).view(1, 1, H, W).float() / 255
    with torch.inference_mode():
        img = (img - DentalDataset.IMG_MEAN) / DentalDataset.IMG_STD
        y_hat = model(img.to(device)).squeeze(0)
        y_hat = torch.sigmoid(y_hat) > 0.5
    unet_mask = y_hat.cpu().numpy()

    h5py_file.create_dataset('segmentation_mask/' + img_name, data=unet_mask, compression='gzip',
                             compression_opts=9)
h5py_file.close()
