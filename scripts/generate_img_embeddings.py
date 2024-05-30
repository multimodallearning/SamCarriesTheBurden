# script to offline generate SAM's image embeddings

from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

from segment_anything import sam_model_registry, SamPredictor

graz_img_dir = Path('data/DentalSeg/img').glob('*.jpg')

# set up SAM
sam_type = ['sam', 'medsam'][0]
print(f'Using {sam_type} model')
if sam_type == 'sam':
    sam_checkpoint = "data/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
elif sam_type == 'medsam':
    sam_checkpoint = "data/medsam_vit_b.pth"
    model_type = "vit_b"

device = "cuda:3" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# create h5 file
h5py_path = Path(f'data/dental_{sam_type}_img_embedding.h5')
h5py_file = h5py.File(h5py_path, 'x')
h5py_file.attrs['checkpoint'] = sam_checkpoint.split('/')[-1]
h5py_file.attrs['img_encoder_img_size'] = sam.image_encoder.img_size
c = 0
for img_file in tqdm(list(graz_img_dir), unit='img', desc='Saving embeddings'):
    c += 1
    # load image
    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # SAM image embedding
    with torch.inference_mode():
        if sam_type == 'sam':
            predictor.set_image(img, image_format="RGB")
            features = predictor.features.cpu().numpy()
            original_size = predictor.original_size
            input_size = predictor.input_size
        elif sam_type == 'medsam':
            img_resize = cv2.resize(
                img,
                (1024, 1024),
                interpolation=cv2.INTER_CUBIC
            )
            # Resizing
            img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8,
                                                                   a_max=None)  # normalize to [0, 1], (H, W, 3
            # convert the shape to (3, H, W)
            assert np.max(img_resize) <= 1.0 and np.min(img_resize) >= 0.0, 'image should be normalized to [0, 1]'
            img_tensor = torch.tensor(img_resize).float().permute(2, 0, 1).unsqueeze(0).to(device)

            features = predictor.model.image_encoder(img_tensor).cpu().numpy()
            original_size = img.shape[:2]
            input_size = img_resize.shape[:2]

    # store to dataset
    curr_group = h5py_file.create_group(f'img_embedding/{img_file.stem}')
    curr_group.attrs['original_size'] = original_size
    curr_group.attrs['input_size'] = input_size
    curr_group.create_dataset('features', data=features, compression='gzip', compression_opts=9)

    predictor.reset_image()

    if c > 10:  # only for testing
        pass
        #break
h5py_file.close()
