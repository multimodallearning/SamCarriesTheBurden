from segment_anything.sam_mask_decoder_head import SAMMaskDecoderHead
from segment_anything.utils.prompt_utils import PromptExtractor
import h5py
import json
import torch
import cv2
from matplotlib import pyplot as plt

sam_checkpoint = "data/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cpu"
img_embedding_h5 = "data/Graz_img_embedding.h5"

sam_predictor = SAMMaskDecoderHead(sam_checkpoint, model_type, device, img_embedding_h5)

img_name = '2639_0713211442_01_WRI-R1_M005'
f = h5py.File('/home/ron/Documents/KidsBoneChecker/datasets/data/GRAZPEDWRI-DX/SegGraz_nnunet_predictions.h5', 'r')
lbl_idx_mapping = json.loads(f.attrs['labels'])
ds_seg_masks = f['nnUNet_prediction']
seg_masks = torch.from_numpy(ds_seg_masks[img_name][:])
prompt_extractor = PromptExtractor(seg_masks)
prompt = prompt_extractor.extract()[-2]

mask, mask_prob, mask_prev_iter = sam_predictor.predict_mask(img_name, prompt, 'box')

image = cv2.imread(f'data/img_only_front_all_left/{img_name}.png', cv2.IMREAD_GRAYSCALE)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

plt.figure()
plt.imshow(image)
plt.imshow(mask.squeeze(), alpha=mask.squeeze().float())
plt.show()
