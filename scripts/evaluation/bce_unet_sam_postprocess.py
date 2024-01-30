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

prompts2use1st = ["box"]
prompts2use2nd = ["pos_points", "neg_points"]
self_refine = True
plot_results = True
use_ccl = True

model_id = '0427c1de20c140c5bff7284c7a4ae614'
cl_model = InputModel(model_id)
model = UNet.load(cl_model.get_weights(), 'cpu').eval()
df = LightSegGrazPedWriDataset('val')

sam_checkpoint = "data/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cpu"
img_embedding_h5 = "data/Graz_img_embedding.h5"
sam_predictor = SAMMaskDecoderHead(sam_checkpoint, model_type, device, img_embedding_h5)

if plot_results:
    dir_name = str.join('_', prompts2use1st)
    if self_refine:
        dir_name += '_self_refine_' + str.join('_', prompts2use2nd)
    dir_name += '' if use_ccl else '_no_ccl'
    plot_save_path = Path('/home/ron/Downloads/SAM_refine_result/UNet/' + dir_name)
    plot_save_path.mkdir(exist_ok=True)
    print(f'plot_save_path: {dir_name}')

dsc_unet = []
dsc_sam = []
dsc_sam_est = []
for img, y, file_name in tqdm(df, unit='img'):
    y = y.unsqueeze(0).bool()
    # forward
    with torch.inference_mode():
        x = (img - df.IMG_MEAN) / df.IMG_STD
        y_hat = model(x.unsqueeze(0)).squeeze(0)
        y_hat = torch.sigmoid(y_hat) > 0.5
    unet_mask = y_hat.clone()

    prompt_extractor = PromptExtractor(unet_mask, use_ccl=use_ccl)
    prompts = prompt_extractor.extract()

    refined_sam_masks = torch.zeros_like(unet_mask, dtype=bool)
    # init with nan to avoid confusion with 0
    est_dice = torch.full((df.N_CLASSES,), float('nan'))
    for prompt in prompts:
        mask, mask_score, mask_prev_iter = sam_predictor.predict_mask(file_name, prompt, prompts2use1st)
        if self_refine:
            mask, mask_score, _ = sam_predictor.predict_mask(file_name, prompt, prompts2use2nd, mask_prev_iter)

        mask = F.interpolate(mask.float(), size=unet_mask.shape[-2:], mode='nearest-exact')
        refined_sam_masks[prompt.class_idx] = mask.squeeze()
        # convert Jaccard to Dice
        est_dice[prompt.class_idx] = 2 * mask_score / (1 + mask_score)

    dsc_unet.append(multilabel_dice(unet_mask.unsqueeze(0), y))
    dsc_sam.append(multilabel_dice(refined_sam_masks.unsqueeze(0), y))
    dsc_sam_est.append(est_dice)

    if plot_results:
        prompt_union = set(prompts2use1st + prompts2use2nd)
        sam_prompt_debug_plots(prompt_extractor, img, unet_mask, refined_sam_masks, est_dice, list(prompt_union),
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
plot_labels = [lbl for lbl, idx in df.BONE_LABEL_MAPPING.items() if nan_mask[idx]]
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
