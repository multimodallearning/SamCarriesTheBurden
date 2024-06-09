# script to offline calculate SAM agreement in dice coefficient

from pathlib import Path

import cv2
import h5py
import torch
from clearml import InputModel
from torch.nn import functional as F
from tqdm import tqdm

from custom_arcitecture.classic_u_net import UNet
from scripts.seg_grazpedwri_dataset import LightSegGrazPedWriDataset
from segment_anything.sam_mask_decoder_head import SAMMaskDecoderHead
from segment_anything.utils.prompt_utils import SAMSelectingPromptExtractor
from utils.dice_coefficient import multilabel_dice

device = "cuda:3" if torch.cuda.is_available() else "cpu"

model_id = 'a7364b31977e42a2a15ac511cfed358f'
print(f'Using model: {model_id}')
cl_model = InputModel(model_id)
model = UNet.load(cl_model.get_weights(), device).eval().to(device)
sam = SAMMaskDecoderHead("data/sam_vit_h_4b8939.pth", "vit_h", device, "data/graz_sam_img_embedding.h5")

H, W = 384, 224
img_dir = Path('data/img_only_front_all_left')

# open h5 file
h5py_path = Path(f'data/seg_masks/{model_id}/raw_segmentations_500.h5')

h5py_file = h5py.File(h5py_path, 'a')
available_files = h5py_file['segmentation_mask'].keys()

for img_name in tqdm(available_files, unit='img', desc='Calculate SAM agreement'):
    img_file = img_dir / (img_name + '.png')
    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    img = torch.from_numpy(img).view(1, 1, H, W).float() / 255
    img = img.to(device)
    with torch.inference_mode():
        img = (img - LightSegGrazPedWriDataset.IMG_MEAN) / LightSegGrazPedWriDataset.IMG_STD
        y_hat = model(img).squeeze(0)
        y_hat = torch.sigmoid(y_hat)
    unet_mask = y_hat.clone()

    # preprocessing
    prompts = SAMSelectingPromptExtractor(unet_mask).extract()
    sam_mask = torch.zeros_like(unet_mask, dtype=torch.bool)
    for prompt in prompts:
        mask, mask_score, mask_prev_iter = sam.predict_mask(img_name, prompt, ['pos_points', 'neg_points'],
                                                             mask_prev_iter=prompt.mask_logits)

        mask = F.interpolate(mask.float(), size=unet_mask.shape[-2:], mode='nearest-exact')
        sam_mask[prompt.class_idx] = mask.squeeze()


    dsc = multilabel_dice(sam_mask.unsqueeze(0), unet_mask.unsqueeze(0) > 0.5).nanmean().item()
    h5py_file['segmentation_mask/' + img_name].attrs['SAM_agreement_dsc'] = dsc
h5py_file.close()
