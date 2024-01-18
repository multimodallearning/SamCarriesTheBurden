from pathlib import Path

import cv2
import h5py
import torch
from tqdm import tqdm

from segment_anything import sam_model_registry, SamPredictor

graz_img_dir = Path('data/img_only_front_all_left').glob('*.png')

# set uo SAM
sam_checkpoint = "data/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cpu"  # "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# create h5 file
h5py_path = Path('data/Graz_img_embedding.h5')
h5py_file = h5py.File(h5py_path, 'x')
h5py_file.attrs['checkpoint'] = sam_checkpoint.split('/')[-1]
h5py_file.attrs['img_encoder_img_size'] = sam.image_encoder.img_size
for img_file in tqdm(list(graz_img_dir), unit='img', desc='Saving embeddings'):
    # load image
    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # SAM image embedding
    with torch.inference_mode():
        predictor.set_image(img, image_format="RGB")

    # store to dataset
    curr_group = h5py_file.create_group(f'img_embedding/{img_file.stem}')
    curr_group.attrs['original_size'] = predictor.original_size
    curr_group.attrs['input_size'] = predictor.input_size
    curr_group.create_dataset('features', data=predictor.features.cpu().numpy(), compression='gzip', compression_opts=9)

    predictor.reset_image()
    break
h5py_file.close()
