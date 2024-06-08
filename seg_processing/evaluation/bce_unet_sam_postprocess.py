from pathlib import Path

import pandas as pd
import seaborn as sns
import torch
from clearml import InputModel
from matplotlib import pyplot as plt
from tqdm import tqdm

from custom_arcitecture.classic_u_net import UNet
from plot_utils import sam_prompt_debug_plots
from scripts.dental_dataset import DentalDataset
from segment_anything.utils.prompt_utils import PromptExtractor
from utils.dice_coefficient import multilabel_dice
from utils.seg_refinement import SAMSegRefiner, SegEnhance

prompts2use1st = ["box"]
prompts2use2nd = ["pos_points", "neg_points"]
plot_results = True

model_id = '274591116b004e348cfe34ac9608ba9e'
cl_model = InputModel(model_id)
model = UNet.load(cl_model.get_weights(), 'cpu').eval()
ds = DentalDataset('val')

sam_type = ['SAM', 'MedSAM'][0]
sam_refiner = SAMSegRefiner(sam_type, 'cpu', [prompts2use1st, prompts2use2nd])
seg_processor = SegEnhance(sam_refiner, 'highest_probability', 'dilation', 'square', 8, 'cpu')

if plot_results:
    dir_name = str.join('_', prompts2use1st)
    if sam_refiner.self_refine:
        dir_name += '_self_refine_' + str.join('_', prompts2use2nd)
    plot_save_path = Path(f'/home/ron/Downloads/SAM_refine_result/{model_id}/{sam_type}/' + dir_name)
    plot_save_path.mkdir(exist_ok=True, parents=True)
    print(f'plot_save_path: {dir_name}')

dsc_unet = []
dsc_sam = []
dsc_sam_est = []
for img, y, file_name in tqdm(ds, unit='img'):
    try:
        if file_name == '0172_0304693626_01_WRI-R1_F014':
            continue
        y = y.unsqueeze(0).bool()
        # forward
        with torch.inference_mode():
            x = (img - ds.IMG_MEAN) / ds.IMG_STD
            y_hat = model(x.unsqueeze(0)).squeeze(0)
            y_hat = torch.sigmoid(y_hat)
        y_hat = y_hat.clone()
        # refinement
        refined_sam_masks, est_dice = seg_processor.enhance(y_hat, file_name)

        unet_mask = y_hat > 0.5
        dsc_unet.append(multilabel_dice(unet_mask.unsqueeze(0), y))
        dsc_sam.append(multilabel_dice(refined_sam_masks.unsqueeze(0).bool(), y))
        dsc_sam_est.append(est_dice)

        if plot_results:
            prompt_union = set(prompts2use1st + prompts2use2nd)
            sam_prompt_debug_plots(PromptExtractor(seg_processor.last_preprocessed_seg > 0.5), img, y_hat, refined_sam_masks,
                                   est_dice, list(prompt_union), plot_save_path / file_name)
    except Exception as e:
        print(f'Skipping. Error in {file_name}: {e}')

dsc_unet = torch.cat(dsc_unet, dim=0)
dsc_sam = torch.cat(dsc_sam, dim=0)
dsc_sam_est = torch.stack(dsc_sam_est, dim=0)

print(f'UNet DSC: {dsc_unet.nanmean()}')
print(f'SAM DSC: {dsc_sam.nanmean()}')

# plotting
# Dice per class
nan_mask = dsc_unet.isnan().all(0) & dsc_sam.isnan().all(0)
nan_mask = ~nan_mask
plot_labels = [lbl for idx, lbl in enumerate(ds.CLASS_LABEL) if nan_mask[idx]]
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
