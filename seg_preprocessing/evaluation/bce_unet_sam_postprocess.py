from pathlib import Path

import pandas as pd
import seaborn as sns
import torch
from clearml import InputModel
from matplotlib import pyplot as plt
from torch.nn import functional as F
from tqdm import tqdm

from plot_utils import sam_prompt_debug_plots
from scripts.seg_grazpedwri_dataset import LightSegGrazPedWriDataset
from segment_anything.sam_mask_decoder_head import SAMMaskDecoderHead
from segment_anything.utils.prompt_utils import PromptExtractor
from unet.classic_u_net import UNet
from utils.dice_coefficient import multilabel_dice
from seg_preprocessing.segmentation_preprocessing import remove_all_but_one_connected_component, dilation

prompts2use1st = ["pos_points"]
prompts2use2nd = ["box"]
self_refine = True
plot_results = True

model_id = ['0427c1de20c140c5bff7284c7a4ae614',  # initial training
            '0ea1c877eedc446b828048456ffd561a',  # sam pseudo labels
            ][0]
cl_model = InputModel(model_id)
model = UNet.load(cl_model.get_weights(), 'cpu').eval()
ds = LightSegGrazPedWriDataset('val')

sam_type = ['SAM', 'MedSAM'][0]
print(f'Using {sam_type} for refinement')
if sam_type == 'SAM':
    sam_checkpoint = "data/sam_vit_h_4b8939.pth"
    sam_model_type = "vit_h"
    img_embedding_h5 = "data/graz_sam_img_embedding.h5"
elif sam_type == 'MedSAM':
    sam_checkpoint = "data/medsam_vit_b.pth"
    sam_model_type = "vit_b"
    img_embedding_h5 = "data/graz_medsam_img_embedding.h5"
else:
    raise NotImplementedError(f'Unknown SAM type: {sam_type}')
sam_predictor = SAMMaskDecoderHead(sam_checkpoint, sam_model_type, 'cpu', img_embedding_h5)

if plot_results:
    dir_name = str.join('_', prompts2use1st)
    if self_refine:
        dir_name += '_self_refine_' + str.join('_', prompts2use2nd)
    plot_save_path = Path(f'/home/ron/Downloads/SAM_refine_result/{model_id}/{sam_type}/' + dir_name)
    plot_save_path.mkdir(exist_ok=True, parents=True)
    print(f'plot_save_path: {dir_name}')

dsc_unet = []
dsc_sam = []
dsc_sam_est = []
for img, y, file_name in tqdm(ds, unit='img'):
    y = y.unsqueeze(0).bool()
    # forward
    with torch.inference_mode():
        x = (img - ds.IMG_MEAN) / ds.IMG_STD
        y_hat = model(x.unsqueeze(0)).squeeze(0)
        y_hat = torch.sigmoid(y_hat)
    y_hat = y_hat.clone()
    # preprocessing
    preprocess_mask = y_hat.clone()
    preprocess_mask = remove_all_but_one_connected_component(preprocess_mask, 'highest_probability', num_iter=250)
    preprocess_mask = preprocess_mask > 0.5
    kernel = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float)
    preprocess_mask = dilation(preprocess_mask.unsqueeze(0).float(), kernel.float(),
                               engine='convolution').squeeze().bool()
    prompt_extractor = PromptExtractor(preprocess_mask)
    prompts = prompt_extractor.extract()

    refined_sam_masks = torch.zeros_like(y_hat, dtype=bool)
    # init with nan to avoid confusion with 0
    est_dice = torch.full((ds.N_CLASSES,), float('nan'))
    for prompt in prompts:
        mask, mask_score, mask_prev_iter = sam_predictor.predict_mask(file_name, prompt, prompts2use1st)
        if self_refine:
            mask, mask_score, _ = sam_predictor.predict_mask(file_name, prompt, prompts2use2nd, mask_prev_iter)

        mask = F.interpolate(mask.float(), size=y_hat.shape[-2:], mode='nearest-exact')
        refined_sam_masks[prompt.class_idx] = mask.squeeze()
        # convert Jaccard to Dice
        est_dice[prompt.class_idx] = 2 * mask_score / (1 + mask_score)

    unet_mask = y_hat > 0.5
    dsc_unet.append(multilabel_dice(unet_mask.unsqueeze(0), y))
    dsc_sam.append(multilabel_dice(refined_sam_masks.unsqueeze(0), y))
    dsc_sam_est.append(est_dice)

    if plot_results:
        prompt_union = set(prompts2use1st + prompts2use2nd)
        sam_prompt_debug_plots(prompt_extractor, img, y_hat, refined_sam_masks, est_dice, list(prompt_union),
                               plot_save_path / file_name)

dsc_unet = torch.cat(dsc_unet, dim=0)
dsc_sam = torch.cat(dsc_sam, dim=0)
dsc_sam_est = torch.stack(dsc_sam_est, dim=0)

print(f'UNet DSC: {dsc_unet.nanmean()}')
print(f'SAM DSC: {dsc_sam.nanmean()}')

# plotting
# Dice per class
nan_mask = dsc_unet.isnan().all(0) & dsc_sam.isnan().all(0)
nan_mask = ~nan_mask
plot_labels = [lbl for lbl, idx in ds.BONE_LABEL_MAPPING.items() if nan_mask[idx]]
plot_dsc_nnunet = dsc_unet[:, nan_mask].nanmean(0).tolist()
plot_dsc_sam = dsc_sam[:, nan_mask].nanmean(0).tolist()
df = pd.DataFrame({'Anatomy': plot_labels,
                   'UNet': plot_dsc_nnunet,
                   'SAM Refinement': plot_dsc_sam})
df = df.melt(id_vars='Anatomy', var_name='Method', value_name='DSC')
plt.figure(figsize=(10, 8))
ax = sns.barplot(x='Anatomy', y='DSC', hue='Method', data=df)
ax.set_xticks(range(len(plot_labels)))
ax.set_xticklabels(plot_labels, rotation=90)
ax.set_title(f'UNet DSC: {dsc_unet.nanmean():.5f}\nSAM DSC: {dsc_sam.nanmean():.5f}')
plt.tight_layout()

if plot_results:
    plt.savefig(plot_save_path / 'dsc.png')

plt.show()

# Difference between estimated and real DSC
plt.figure(figsize=(10, 8))
diff = dsc_sam_est - dsc_sam
diff = diff[:, nan_mask].nanmean(0).tolist()
sns.barplot(x=plot_labels, y=diff)
plt.title('Difference between estimated and real DSC')
plt.xticks(rotation=90)

if plot_results:
    plt.savefig(plot_save_path / 'dsc_diff.png')

plt.show()

# Plot distribution of estimated DSC
plt.figure(figsize=(10, 8))
sns.histplot(dsc_sam_est.flatten(), stat='percent')
plt.title('Distribution of estimated DSC')
plt.xlabel('DSC')

if plot_results:
    plt.savefig(plot_save_path / 'dist_est_dsc.png')

plt.show()
