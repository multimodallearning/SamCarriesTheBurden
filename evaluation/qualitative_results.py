import pandas as pd
import torch
from clearml import InputModel
from matplotlib import pyplot as plt

import clearml_model_id
from custom_arcitecture.classic_u_net import UNet
from custom_arcitecture.lraspp import LRASPPOnSAM
from scripts.seg_grazpedwri_dataset import LightSegGrazPedWriDataset
from utils.dice_coefficient import multilabel_dice

# parameters
architecture = 'UNet_pseudo_lbl_sam'
num_train_samples = 43
file_choice = 'best'
print(f'Architecture: {architecture}, num_train_samples: {num_train_samples}')

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

df_dsc = pd.read_csv(f'evaluation/csv_results/{architecture}_raw_dsc.csv')
df_dsc = df_dsc[df_dsc['Number training samples'] == num_train_samples]
df_dsc = df_dsc[df_dsc['File stem'] != '0172_0304693626_01_WRI-R1_F014']
# get file_stem
if file_choice == 'median':
    file2plot = df_dsc.loc[df_dsc['DSC mean'].quantile(q=0.5, interpolation='nearest') == df_dsc['DSC mean'], 'File stem'].values[0]
elif file_choice == 'worst':
    file2plot = df_dsc.loc[df_dsc['DSC mean'].idxmin(), 'File stem']
elif file_choice == 'best':
    file2plot = df_dsc.loc[df_dsc['DSC mean'].idxmax(), 'File stem']
else:
    raise ValueError('Unknown file choice')
idx2plot = ds.available_file_names.index(file2plot)
print(f'File to plot: {file2plot}')

img, y, file_stem = ds[idx2plot]
y = y.unsqueeze(0).bool()
# forward
with torch.inference_mode():
    x = (img - ds.IMG_MEAN) / ds.IMG_STD
    y_hat = model(x.unsqueeze(0))
    y_hat = torch.sigmoid(y_hat) > 0.5

dsc = multilabel_dice(y_hat, y).squeeze()

img = img.squeeze()
y_hat = y_hat.squeeze()
y = y.squeeze()

# overview figure
fig, ax = plt.subplots(1, 3)
ax[0].imshow(img, 'gray')
ax[0].set_title('Input image')
ax[1].imshow(img, 'gray')
ax[1].imshow(y.float().argmax(0), alpha=y.any(0).float() * .8, cmap='tab20', interpolation='nearest')
ax[1].set_title('Ground truth')
ax[2].imshow(img, 'gray')
ax[2].imshow(y_hat.float().argmax(0), alpha=y_hat.any(0).float() * .8, cmap='tab20', interpolation='nearest')
ax[2].set_title('Prediction')
fig.suptitle(f'DSC: {dsc.nanmean().item():.3f}')
fig.tight_layout()

# iterate over all classes
for lbl, lbl_idx in ds.BONE_LABEL_MAPPING.items():
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img, 'gray')
    ax[0].set_title('Input image')
    ax[1].imshow(img, 'gray')
    ax[1].imshow(y[lbl_idx], alpha=y[lbl_idx].float())
    ax[1].set_title('Ground truth')
    ax[2].imshow(img, 'gray')
    ax[2].imshow(y_hat[lbl_idx], alpha=y_hat[lbl_idx].float())
    ax[2].set_title('Prediction')
    fig.suptitle(f'{file_stem}\n{lbl} DSC: {dsc[lbl_idx].item():.3f}')
    fig.tight_layout()

plt.show()
