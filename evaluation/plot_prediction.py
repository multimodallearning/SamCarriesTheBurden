# Plot ground truth and prediction of specific file also save the DSC in a text file.

from pathlib import Path

import numpy as np
import torch
from clearml import InputModel
from matplotlib import pyplot as plt

import clearml_model_id
from custom_arcitecture.classic_u_net import UNet
from custom_arcitecture.lraspp import LRASPPOnSAM
from scripts.seg_grazpedwri_dataset import LightSegGrazPedWriDataset
from utils.dice_coefficient import multilabel_dice

# parameters
architecture = 'UNet_mean_teacher'
num_train_samples = 43
print(f'Architecture: {architecture}, num_train_samples: {num_train_samples}')
save_dir = Path(f'/home/ron/Desktop/plots/{architecture}')
save_dir.mkdir(exist_ok=True, parents=True)

ds = LightSegGrazPedWriDataset('test')

model_id = {
    'UNet': clearml_model_id.unet_ids,
    'UNet_pseudo_lbl_raw': clearml_model_id.raw_pseudo_lbl_unet_ids,
    'UNet_pseudo_lbl_sam': clearml_model_id.sam_pseudo_lbl_unet_ids,
    'SAM_LRASPP': clearml_model_id.sam_lraspp,
    'UNet_mean_teacher': clearml_model_id.unet_mean_teacher_ids
}[architecture][num_train_samples]

cl_model = InputModel(model_id)
if architecture.startswith('UNet'):
    model = UNet.load(cl_model.get_weights(), 'cpu').eval()
elif architecture.startswith('SAM'):
    model = LRASPPOnSAM.load(cl_model.get_weights(), 'cpu').eval()
else:
    raise ValueError('Unknown architecture.')

file2plot = '0476_0834388162_01_WRI-R1_F005'
idx2plot = ds.available_file_names.index(file2plot)

img, y, file_stem = ds[idx2plot]
y = y.unsqueeze(0).bool()
# forward
with torch.inference_mode():
    x = (img - ds.IMG_MEAN) / ds.IMG_STD
    y_hat = model(x.unsqueeze(0))
    y_hat = torch.sigmoid(y_hat) > 0.5

dsc = multilabel_dice(y_hat, y).squeeze()
print('DSC:', dsc.nanmean().item())

img = img.squeeze()
y_hat = y_hat.squeeze()
y = y.squeeze()

# overview figure
plt.figure()
plt.imshow(img, 'gray')
plt.axis('off')
plt.tight_layout()
plt.gcf().savefig(save_dir / 'image.png', dpi=400)

plt.figure()
plt.imshow(img, 'gray')
plt.imshow(y.float().argmax(0), alpha=y.any(0).float() * .6, cmap='tab20', interpolation='nearest')
plt.axis('off')
plt.tight_layout()
plt.gcf().savefig(save_dir / 'gt.png', dpi=400)

plt.figure()
plt.imshow(img, 'gray')
plt.imshow(y_hat.float().argmax(0), alpha=y_hat.any(0).float() * .6, cmap='tab20', interpolation='nearest')
plt.axis('off')
plt.tight_layout()
plt.gcf().savefig(save_dir / f'{architecture}.png', dpi=400)

# save dsc
with open(save_dir / 'dsc.txt', 'w') as f:
    f.write(f'DSC: {dsc.nanmean().item()} +/- {np.nanstd(dsc).item()}')

plt.show()
