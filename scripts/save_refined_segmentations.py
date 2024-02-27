import json
from pathlib import Path

import cv2
import h5py
import pandas as pd
import torch
from clearml import InputModel
from tqdm import tqdm

from custom_arcitecture.classic_u_net import UNet
from scripts.seg_grazpedwri_dataset import LightSegGrazPedWriDataset
from utils.seg_refinement import SegEnhance, SAMSegRefiner

device = "cuda:4" if torch.cuda.is_available() else "cpu"

model_id = '29f483c5ab6d4f2991f96958d6c68b1a'
cl_model = InputModel(model_id)
model = UNet.load(cl_model.get_weights(), device).eval().to(device)
H, W = 384, 224

refine_params = {
    'prompts2use': [["box"], ["pos_points", "neg_points"]],
    'ccl_selection': 'highest_probability',
    'morph_op': 'dilation',
    'struct_elem': 'square',
    'radius': 8,
}
sam_refiner = SAMSegRefiner('SAM', device, refine_params['prompts2use'])
seg_processor = SegEnhance(sam_refiner, refine_params['ccl_selection'], refine_params['morph_op'],
                           refine_params['struct_elem'], refine_params['radius'], device)

print(f'Refine model {model_id} segmentation with {refine_params}')

n_files = 500
img_dir = Path('data/img_only_front_all_left')
available_files = pd.read_csv(f'data/{n_files}unlabeled_sample.csv', index_col='filestem').index.tolist()

# create h5 file
h5py_path = Path(f'data/seg_masks/{model_id}')
h5py_path.mkdir(parents=True, exist_ok=True)
id_str = str.join('_', refine_params['prompts2use'][0]) + '_refine_' + str.join('_', refine_params['prompts2use'][1])
h5py_path = h5py_path / f'sam_{id_str}_{n_files}.h5'

h5py_file = h5py.File(h5py_path, 'w')
# store labels and their index
h5py_file.attrs['labels'] = json.dumps(LightSegGrazPedWriDataset.BONE_LABEL_MAPPING)
h5py_file.attrs['refine_params'] = json.dumps(refine_params)

for img_name in tqdm(available_files, unit='img', desc='Refine segmentation'):
    img_file = img_dir / (img_name + '.png')
    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    img = torch.from_numpy(img).view(1, 1, H, W).float() / 255
    with torch.inference_mode():
        img = (img - LightSegGrazPedWriDataset.IMG_MEAN) / LightSegGrazPedWriDataset.IMG_STD
        y_hat = model(img.to(device)).squeeze(0)
        y_hat = torch.sigmoid(y_hat)
    unet_mask = y_hat.clone()

    # preprocessing
    refined_sam_masks, est_dice = seg_processor.enhance(y_hat, img_name)

    refined_sam_masks = refined_sam_masks.cpu().numpy()
    est_dice = est_dice.cpu().numpy()

    h5py_file.create_dataset('segmentation_mask/' + img_name, data=refined_sam_masks, compression='gzip',
                             compression_opts=9)
    h5py_file['segmentation_mask/' + img_name].attrs['estimated_dice'] = est_dice
h5py_file.close()
