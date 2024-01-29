import json
from pathlib import Path

import cv2
import h5py
import torch
from clearml import InputModel
from torch.nn import functional as F
from tqdm import tqdm

from scripts.seg_grazpedwri_dataset import LightSegGrazPedWriDataset
from segment_anything.sam_mask_decoder_head import SAMMaskDecoderHead
from segment_anything.utils.prompt_utils import PromptExtractor
from unet.classic_u_net import UNet

device = "cuda:3" if torch.cuda.is_available() else "cpu"

model_id = '404bd577195044749a1658ecd76912f7'
cl_model = InputModel(model_id)
model = UNet.load(cl_model.get_weights(), device).eval().to(device)
H, W = 384, 224

sam_checkpoint = "data/sam_vit_h_4b8939.pth"
model_type = "vit_h"
img_embedding_h5 = "data/Graz_img_embedding.h5"
sam_predictor = SAMMaskDecoderHead(sam_checkpoint, model_type, device, img_embedding_h5)

# create h5 file
h5py_path = Path(f'data/seg_masks/sam_{model_id}.h5')
img_dir = Path('data/img_only_front_all_left').glob('*.png')

h5py_file = h5py.File(h5py_path, 'w')
# store labels and their index
h5py_file.attrs['labels'] = json.dumps(LightSegGrazPedWriDataset.BONE_LABEL_MAPPING)

for img_file in tqdm(list(img_dir), unit='img', desc='Refine segmentation'):
    img_name = img_file.stem
    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    img = torch.from_numpy(img).view(1, 1, H, W).float() / 255
    with torch.inference_mode():
        img = (img - LightSegGrazPedWriDataset.IMG_MEAN) / LightSegGrazPedWriDataset.IMG_STD
        y_hat = model(img.to(device)).squeeze(0)
        y_hat = torch.sigmoid(y_hat) > 0.5
    unet_mask = y_hat.clone()

    prompt_extractor = PromptExtractor(unet_mask, use_ccl=True)
    prompts = prompt_extractor.extract()

    refined_sam_masks = torch.zeros_like(unet_mask, dtype=bool)
    for prompt in prompts:
        _, _, mask_prev_iter = sam_predictor.predict_mask(img_name, prompt, "box")
        mask, mask_score, _ = sam_predictor.predict_mask(img_name, prompt, ["pos_points", "neg_points"], mask_prev_iter)

        mask = F.interpolate(mask.float(), size=unet_mask.shape[-2:], mode='nearest-exact')
        refined_sam_masks[prompt.class_idx] = mask.squeeze()
    refined_sam_masks = refined_sam_masks.cpu().numpy()

    h5py_file.create_dataset('segmentation_mask/' + img_name, data=refined_sam_masks, compression='gzip',
                             compression_opts=9)
h5py_file.close()
