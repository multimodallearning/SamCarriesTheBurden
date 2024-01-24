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
prompts2use2nd = ["box"]
self_refine = True
plot_results = True

model_id = '404bd577195044749a1658ecd76912f7'
cl_model = InputModel(model_id)
model = UNet.load(cl_model.get_weights(), 'cpu').eval()
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
    plot_save_path = Path('/home/ron/Downloads/SAM_refine_result/UNet/' + dir_name)
    plot_save_path.mkdir(exist_ok=True)
    print(f'plot_save_path: {dir_name}')

dsc_unet = []
dsc_sam = []
for img, y, file_name in tqdm(ds, unit='img'):
    y = y.unsqueeze(0).bool()
    # forward
    with torch.inference_mode():
        x = (img - ds.IMG_MEAN) / ds.IMG_STD
        y_hat = model(x.unsqueeze(0)).squeeze(0)
        y_hat = torch.sigmoid(y_hat) > 0.5
    unet_mask = y_hat.clone()

    prompt_extractor = PromptExtractor(unet_mask, use_ccl=True)
    prompts = prompt_extractor.extract()

    refined_sam_masks = torch.zeros_like(unet_mask, dtype=bool)
    for prompt in prompts:
        mask, mask_score, mask_prev_iter = sam_predictor.predict_mask(file_name, prompt, prompts2use1st)
        if self_refine:
            mask, mask_score, _ = sam_predictor.predict_mask(file_name, prompt, prompts2use2nd, mask_prev_iter)

        mask = F.interpolate(mask.float(), size=unet_mask.shape[-2:], mode='nearest-exact')
        refined_sam_masks[prompt.class_idx] = mask.squeeze()

    dsc_unet.append(multilabel_dice(unet_mask.unsqueeze(0), y))
    dsc_sam.append(multilabel_dice(refined_sam_masks.unsqueeze(0), y))

    if plot_results:
        prompt_union = set(prompts2use1st + prompts2use2nd)
        sam_prompt_debug_plots(prompt_extractor, img, unet_mask, refined_sam_masks, list(prompt_union),
                               plot_save_path / file_name)

dsc_unet = torch.cat(dsc_unet, dim=0)
dsc_sam = torch.cat(dsc_sam, dim=0)

print(f'UNet DSC: {dsc_unet.nanmean()}')
print(f'SAM DSC: {dsc_sam.nanmean()}')

# plotting
nan_mask = dsc_unet.isnan().all(0) & dsc_sam.isnan().all(0)
nan_mask = ~nan_mask
plot_labels = [lbl for lbl, idx in ds.BONE_LABEL_MAPPING.items() if nan_mask[idx]]
plot_dsc_nnunet = dsc_unet[:, nan_mask].nanmean(0).tolist()
plot_dsc_sam = dsc_sam[:, nan_mask].nanmean(0).tolist()
ds = pd.DataFrame({'Anatomy': plot_labels,
                   'UNet': plot_dsc_nnunet,
                   'SAM Refinement': plot_dsc_sam})
ds = ds.melt(id_vars='Anatomy', var_name='Method', value_name='DSC')
plt.figure(figsize=(10, 8))
ax = sns.barplot(x='Anatomy', y='DSC', hue='Method', data=ds)
ax.set_xticks(range(len(plot_labels)))
ax.set_xticklabels(plot_labels, rotation=90)
ax.set_title(f'UNet DSC: {dsc_unet.nanmean():.5f}\nSAM DSC: {dsc_sam.nanmean():.5f}')
plt.tight_layout()

if plot_results:
    plt.savefig(plot_save_path / 'dsc.png')

plt.show()
