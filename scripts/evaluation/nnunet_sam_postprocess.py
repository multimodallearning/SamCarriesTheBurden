import json
from pathlib import Path

import h5py
import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F
from tqdm import tqdm

from scripts.seg_grazpedwri_dataset import LightSegGrazPedWriDataset
from segment_anything.sam_mask_decoder_head import SAMMaskDecoderHead
from segment_anything.utils.prompt_utils import PromptExtractor
from utils.dice_coefficient import multilabel_dice
import seaborn as sns
import pandas as pd
from plot_utils import sam_prompt_debug_plots

prompts2use1st = ["box"]
prompts2use2nd = ["box"]
self_refine = False
plot_results = True

f = h5py.File('data/SegGraz_nnunet_predictions.h5', 'r')
lbl_idx_mapping = json.loads(f.attrs['labels'])
ds_seg_masks = f['nnUNet_prediction']
ds = LightSegGrazPedWriDataset('test')

sam_checkpoint = "data/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cpu"
img_embedding_h5 = "data/Graz_img_embedding.h5"
sam_predictor = SAMMaskDecoderHead(sam_checkpoint, model_type, device, img_embedding_h5)

if plot_results:
    dir_name = str.join('_', prompts2use1st)
    if self_refine:
        dir_name += '_self_refine_' + str.join('_', prompts2use2nd)
    plot_save_path = Path('/home/ron/Downloads/SAM_refine_result/nnUNet/' + dir_name)
    plot_save_path.mkdir(exist_ok=True)
    print(f'plot_save_path: {dir_name}')

dsc_nnunet = []
dsc_sam = []
for img, y, file_name in tqdm(ds, unit='img'):
    y = y.unsqueeze(0).bool()
    full_nnunet_mask = torch.from_numpy(ds_seg_masks[file_name][:])
    nnunet_mask = []
    for lbl in ds.BONE_LABEL:
        nnunet_mask.append(full_nnunet_mask[lbl_idx_mapping[lbl]])
    nnunet_mask = torch.stack(nnunet_mask, dim=0)

    prompt_extractor = PromptExtractor(nnunet_mask, use_ccl=True)
    prompts = prompt_extractor.extract()

    refined_sam_masks = torch.zeros_like(nnunet_mask, dtype=bool)
    for prompt in prompts:
        mask, mask_score, mask_prev_iter = sam_predictor.predict_mask(file_name, prompt, prompts2use1st)
        if self_refine:
            mask, mask_score, _ = sam_predictor.predict_mask(file_name, prompt, prompts2use2nd, mask_prev_iter)

        mask = F.interpolate(mask.float(), size=nnunet_mask.shape[-2:], mode='nearest-exact')
        refined_sam_masks[prompt.class_idx] = mask.squeeze()

    dsc_nnunet.append(multilabel_dice(nnunet_mask.unsqueeze(0), y))
    dsc_sam.append(multilabel_dice(refined_sam_masks.unsqueeze(0), y))

    if plot_results:
        prompt_union = set(prompts2use1st + prompts2use2nd)
        sam_prompt_debug_plots(prompt_extractor, img, nnunet_mask, refined_sam_masks, list(prompt_union),
                               plot_save_path / file_name)

dsc_nnunet = torch.cat(dsc_nnunet, dim=0)
dsc_sam = torch.cat(dsc_sam, dim=0)

print(f'nnUNet DSC: {dsc_nnunet.nanmean()}')
print(f'SAM DSC: {dsc_sam.nanmean()}')

# plotting
nan_mask = dsc_nnunet.isnan().all(0) & dsc_sam.isnan().all(0)
nan_mask = ~nan_mask
plot_labels = [lbl for lbl, idx in ds.BONE_LABEL_MAPPING.items() if nan_mask[idx]]
plot_dsc_nnunet = dsc_nnunet[:, nan_mask].nanmean(0).tolist()
plot_dsc_sam = dsc_sam[:, nan_mask].nanmean(0).tolist()
ds = pd.DataFrame({'Anatomy': plot_labels,
                   'nnUNet': plot_dsc_nnunet,
                   'SAM Refinement': plot_dsc_sam})
ds = ds.melt(id_vars='Anatomy', var_name='Method', value_name='DSC')
plt.figure(figsize=(10, 8))
ax = sns.barplot(x='Anatomy', y='DSC', hue='Method', data=ds)
ax.set_xticks(range(len(plot_labels)))
ax.set_xticklabels(plot_labels, rotation=90)
ax.set_title(f'nnUNet DSC: {dsc_nnunet.nanmean():.5f}\nSAM DSC: {dsc_sam.nanmean():.5f}')
plt.tight_layout()

if plot_results:
    plt.savefig(plot_save_path / 'dsc.png')

plt.show()
