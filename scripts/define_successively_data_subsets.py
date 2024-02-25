from scripts.seg_grazpedwri_dataset import LightSegGrazPedWriDataset
import torch
from matplotlib import pyplot as plt
import pandas as pd
import random

ds = LightSegGrazPedWriDataset('train')
files = ds.available_file_names
seg_mask = torch.stack([ds.data[f]['mask'] for f in files])
seg_mask = seg_mask.sum((-2, -1))
all_classes_present = seg_mask.all(1)
for idx in torch.where(all_classes_present)[0]:
    f_name = files[idx]
    img = ds.data[f_name]['image'].squeeze()
    mask = ds.data[f_name]['mask']

    fig, axs = plt.subplots(1, 2)
    fig.suptitle(f_name)
    axs[0].imshow(img, 'gray')
    axs[1].imshow(img, 'gray')
    axs[1].imshow(mask.argmax(0), alpha=mask.any(0).float())
selected_index = 13  # manually selected index after visual inspection
print('selected file:', files[selected_index])

files_ordered_for_successively_training = files[:]
del files_ordered_for_successively_training[selected_index]
files_ordered_for_successively_training.insert(0, files[selected_index])
series_files_subset = pd.Series(data=files_ordered_for_successively_training, name='file_stem')
assert series_files_subset.is_unique, 'files are not unique'
assert len(series_files_subset) == len(ds), 'files are missing or duplicated'

# save to file
series_files_subset.to_csv('data/successively_training_files_order.csv', header=True)
