from pathlib import Path

import pandas as pd
import seaborn as sns
import torch
from clearml import InputModel
from matplotlib import pyplot as plt
from tqdm import tqdm

from plot_utils import plot_rnd_walk
from scripts.seg_grazpedwri_dataset import LightSegGrazPedWriDataset
from unet.classic_u_net import UNet
from utils.dice_coefficient import multilabel_dice
from utils.random_walk import random_walk
from utils import segmentation_preprocessing
from kornia.morphology import erosion
from skimage.morphology import disk

plot_results = True

model_id = '0427c1de20c140c5bff7284c7a4ae614'  # initial training
cl_model = InputModel(model_id)
model = UNet.load(cl_model.get_weights(), 'cpu').eval()
ds = LightSegGrazPedWriDataset('val')

if plot_results:
    plot_save_path = Path(f'/home/ron/Downloads/RndWalk/{model_id}')
    plot_save_path.mkdir(exist_ok=True, parents=True)

dsc_unet = []
dsc_rnd_walk = []
for img, y, file_name in tqdm(ds, unit='img'):
    y = y.unsqueeze(0).bool()
    # forward
    with torch.inference_mode():
        x = (img - ds.IMG_MEAN) / ds.IMG_STD
        y_hat = model(x.unsqueeze(0)).squeeze(0)
        y_hat = torch.sigmoid(y_hat) > 0.5
    unet_mask = y_hat.clone()

    # preprocess mask
    preproc_mask = unet_mask.clone()
    erosion_kernel = torch.from_numpy(disk(4, dtype=int))
    preproc_mask = segmentation_preprocessing.remove_all_but_largest_connected_component(preproc_mask, num_iter=250)
    preproc_mask = erosion(preproc_mask.unsqueeze(0).float(), kernel=erosion_kernel).squeeze().bool()

    img = img.squeeze(0)
    p_hat_rnd_walk = random_walk(img, preproc_mask)
    refined_mask = p_hat_rnd_walk > 0.5

    dsc_unet.append(multilabel_dice(unet_mask.unsqueeze(0), y))
    dsc_rnd_walk.append(multilabel_dice(refined_mask.unsqueeze(0), y))

    if plot_results:
        plot_rnd_walk(img, unet_mask, preproc_mask, p_hat_rnd_walk, plot_save_path / file_name)

dsc_unet = torch.cat(dsc_unet, dim=0)
dsc_rnd_walk = torch.cat(dsc_rnd_walk, dim=0)

print(f'UNet DSC: {dsc_unet.nanmean()}')
print(f'Random Walk DSC: {dsc_rnd_walk.nanmean()}')

# plotting
# Dice per class
nan_mask = dsc_unet.isnan().all(0) & dsc_rnd_walk.isnan().all(0)
nan_mask = ~nan_mask
plot_labels = [lbl for lbl, idx in ds.BONE_LABEL_MAPPING.items() if nan_mask[idx]]
plot_dsc_nnunet = dsc_unet[:, nan_mask].nanmean(0).tolist()
plot_dsc_sam = dsc_rnd_walk[:, nan_mask].nanmean(0).tolist()
df = pd.DataFrame({'Anatomy': plot_labels,
                   'UNet': plot_dsc_nnunet,
                   'RndWalk Refinement': plot_dsc_sam})
df = df.melt(id_vars='Anatomy', var_name='Method', value_name='DSC')
plt.figure(figsize=(10, 8))
ax = sns.barplot(x='Anatomy', y='DSC', hue='Method', data=df)
ax.set_xticks(range(len(plot_labels)))
ax.set_xticklabels(plot_labels, rotation=90)
ax.set_title(f'UNet DSC: {dsc_unet.nanmean():.5f}\nRndWalk DSC: {dsc_rnd_walk.nanmean():.5f}')
plt.tight_layout()

if plot_results:
    plt.savefig(plot_save_path / 'dsc.png')

plt.show()
