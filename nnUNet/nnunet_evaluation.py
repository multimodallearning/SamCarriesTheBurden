import torch
from tqdm import tqdm

from nnUNet.prediction_loader import NNUNetPredictionLoader
from scripts.seg_grazpedwri_dataset import LightSegGrazPedWriDataset
from utils.dice_coefficient import multilabel_dice

ds = LightSegGrazPedWriDataset('val')
nnunet_prediction = NNUNetPredictionLoader()

dsc = []
for _, y, file in tqdm(ds, desc='Evaluating'):
    p_hat = nnunet_prediction[file]
    y_hat = p_hat > 0.5

    dsc.append(multilabel_dice(y_hat.unsqueeze(0), y.bool().unsqueeze(0)))

dsc = torch.cat(dsc)
print('#### Dice ####')
for lbl, lbl_idx in ds.BONE_LABEL_MAPPING.items():
    print(f'{lbl}: {dsc[:, lbl_idx].nanmean().item()}')

print(f'\nAverage: {dsc.nanmean().item()}')
