# Evaluates the nnUNet model with different number of training samples

import pandas as pd
import torch

from nnUNet.prediction_loader import NNUNetPredictionLoader
from scripts.dental_dataset import DentalDataset
from utils.dice_coefficient import multilabel_dice

ds = DentalDataset('test')
df = pd.DataFrame()
nnunet_prediction = NNUNetPredictionLoader()

dsc = []
for _, y, file_stem in ds:
    p_hat = nnunet_prediction[file_stem]
    y_hat = p_hat > 0.5

    dsc.append(multilabel_dice(y_hat.unsqueeze(0), y.bool().unsqueeze(0)))
    value_dict = {
        'Method': 'nnUNet',
        'DSC mean': dsc[-1].nanmean().item(),
        'File stem': file_stem
    }
    value_dict.update({'DSC ' + lbl: dsc[-1][0][i].item() for i, lbl in enumerate(ds.CLASS_LABEL)})
    df = pd.concat([df, pd.DataFrame(value_dict, index=[len(df)])], ignore_index=True)

dsc = torch.cat(dsc)
print(f'nnUNet: {dsc.nanmean().item()}')
df.to_csv('evaluation/csv_results/dental/nnunet.csv', index=False)
