# script to offline calculate SAM agreement in dice coefficient

from pathlib import Path

import cv2
import h5py
import torch
from clearml import InputModel
from torch.nn import functional as F
from tqdm import tqdm

from custom_arcitecture.classic_u_net import UNet
from dental_dataset import DentalDataset
from evaluation.clearml_model_id import dental_models
from segment_anything.sam_mask_decoder_head import SAMMaskDecoderHead
from segment_anything.utils.prompt_utils import SAMSelectingPromptExtractor
from utils.dice_coefficient import multilabel_dice
from matplotlib import pyplot as plt

device = "cuda:3" if torch.cuda.is_available() else "cpu"

model_id = dental_models['mean_teacher']
cl_model = InputModel(model_id)
model = UNet.load(cl_model.get_weights(), device).eval().to(device)
sam = SAMMaskDecoderHead("data/sam_vit_h_4b8939.pth", "vit_h", device, "data/dental_sam_img_embedding.h5")

H, W = 224, 384
img_dir = Path('data/DentalSeg/img')

# open h5 file
h5py_path = Path(f'data/seg_masks/{model_id}/raw_segmentations_450.h5')

h5py_file = h5py.File(h5py_path, 'a')
available_files = h5py_file['segmentation_mask'].keys()

for img_name in tqdm(available_files, unit='img', desc='Calculate SAM agreement'):
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
    prompts = SAMSelectingPromptExtractor(unet_mask).extract()
    sam_mask = torch.zeros_like(unet_mask, dtype=torch.bool)
    for prompt in prompts:
        mask, mask_score, mask_prev_iter = sam.predict_mask(img_name, prompt, ['pos_points', 'neg_points'],
                                                             mask_prev_iter=prompt.mask_logits)

        mask = F.interpolate(mask.float(), size=unet_mask.shape[-2:], mode='nearest-exact')
        sam_mask[prompt.class_idx] = mask.squeeze()


    dsc = multilabel_dice(sam_mask.unsqueeze(0), unet_mask.unsqueeze(0) > 0.5).nanmean().item()
    print(f'{img_name} SAM agreement: {dsc}')
    #h5py_file['segmentation_mask/' + img_name].attrs['SAM_agreement_dsc'] = dsc
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(unet_mask.argmax(0).cpu(), alpha=unet_mask.bool().any(0).float(), cmap='jet')
    axs[0].set_title('UNet')

    axs[1].imshow(sam_mask.float().argmax(0).cpu(), alpha=sam_mask.any(0).float(), cmap='jet')
    axs[1].set_title('SAM')
    plt.show()
h5py_file.close()
