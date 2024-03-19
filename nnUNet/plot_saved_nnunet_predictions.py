# script to plot the saved nnUNet predictions

from matplotlib import pyplot as plt

from nnUNet.prediction_loader import NNUNetPredictionLoader
from scripts.seg_grazpedwri_dataset import LightSegGrazPedWriDataset
from random import randint

ds = LightSegGrazPedWriDataset('val')
nnunet_prediction = NNUNetPredictionLoader()

img, y, file = ds[0]
p_hat = nnunet_prediction[file]
y_hat = p_hat > 0.5
for lbl, lbl_idx in ds.BONE_LABEL_MAPPING.items():
    if not y_hat[lbl_idx].any():
        continue

    plt.figure(lbl)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.imshow(p_hat[lbl_idx], alpha=y_hat[lbl_idx].float())
    plt.title(lbl)
plt.show()
