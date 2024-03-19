# Evaluates the nnUNet model with different number of training samples

import torch
from tqdm import tqdm

from nnUNet.prediction_loader import NNUNetPredictionLoader
from scripts.seg_grazpedwri_dataset import LightSegGrazPedWriDataset
from utils.dice_coefficient import multilabel_dice
import pandas as pd

ds = LightSegGrazPedWriDataset('test')
df = pd.DataFrame()
for num_train in tqdm([1, 5, 10, 15, 20, 25, 30, 35, 43]):
    nnunet_prediction = NNUNetPredictionLoader(num_train_samples=num_train)

    dsc = []
    for _, y, file_stem in ds:
        p_hat = nnunet_prediction[file_stem]
        y_hat = p_hat > 0.5

        dsc.append(multilabel_dice(y_hat.unsqueeze(0), y.bool().unsqueeze(0)))
        value_dict = {
            'Method': 'nnUNet',
            'Number training samples': num_train,
            'DSC mean': dsc[-1].nanmean().item(),
            'File stem': file_stem
        }
        value_dict.update({'DSC ' + lbl: dsc[-1][0][i].item() for lbl, i in ds.BONE_LABEL_MAPPING.items()})
        df = pd.concat([df, pd.DataFrame(value_dict, index=[len(df)])], ignore_index=True)

    dsc = torch.cat(dsc)
    print(f'nnUNet {num_train} samples: {dsc.nanmean().item()}')
df.to_csv('evaluation/csv_results/nnunet.csv', index=False)
