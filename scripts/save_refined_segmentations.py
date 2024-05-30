# script to offline generate segmentation masks with SAM's refinement

import json
from pathlib import Path

import cv2
import h5py
import torch
from clearml import InputModel
from tqdm import tqdm

from custom_arcitecture.classic_u_net import UNet
from dental_dataset import DentalDataset
from utils.seg_refinement import SegEnhance, SAMSegRefiner

device = "cuda:3" if torch.cuda.is_available() else "cpu"

model_id = 'fff060f575994796936422b8c2819c5e'
cl_model = InputModel(model_id)
model = UNet.load(cl_model.get_weights(), device).eval().to(device)
H, W = 224, 384

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

img_dir = Path('data/DentalSeg/img')
available_files = DentalDataset('train').available_files

# create h5 file
h5py_path = Path(f'data/seg_masks/{model_id}')
h5py_path.mkdir(parents=True, exist_ok=True)
id_str = str.join('_', refine_params['prompts2use'][0]) + '_refine_' + str.join('_', refine_params['prompts2use'][1])
h5py_path = h5py_path / f'sam_{id_str}_{len(available_files)}.h5'

h5py_file = h5py.File(h5py_path, 'w')
# store labels and their index
h5py_file.attrs['labels'] = json.dumps({lbl:idx for idx, lbl in enumerate(DentalDataset.CLASS_LABEL)})
h5py_file.attrs['refine_params'] = json.dumps(refine_params)
h5py_file.attrs['clearml_model_id'] = model_id

for img_name in tqdm(available_files, unit='img', desc='Refine segmentation'):
    img_file = img_dir / (img_name + '.jpg')
    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    img = torch.from_numpy(img).view(1, 1, H, W).float() / 255
    img = img.to(device)
    with torch.inference_mode():
        img = (img - DentalDataset.IMG_MEAN) / DentalDataset.IMG_STD
        y_hat = model(img).squeeze(0)
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
