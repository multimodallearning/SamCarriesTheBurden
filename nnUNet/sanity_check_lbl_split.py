from matplotlib import pyplot as plt
from pathlib import Path
from random import randint
from numpy import ma

img_dir = Path('/home/ron/Downloads/nnUNet_datasets/imagesTr').glob('*.png')
lbl_dir = '/home/ron/Downloads/nnUNet_datasets'
img_files = sorted(list(img_dir))
n_splits = len(list(Path(lbl_dir).glob('Dataset04*')))

idx = randint(0, len(img_files) - 1)
img = plt.imread(img_files[idx])
fig, axs = plt.subplots(1, n_splits + 1, figsize=(10, 5))
axs = axs.flatten()
assert len(axs) == n_splits + 1
axs[0].imshow(img, 'gray')

img_name = img_files[idx].stem[:-5] + '.png'
for split in range(n_splits):
    split_lbl_dir = f'Dataset{40 + split:03d}_Light_Seg_Graz_split{split}_mutex_all_left'
    lbl = plt.imread(Path(lbl_dir).joinpath(split_lbl_dir, 'labelsTr', img_name))
    axs[split + 1].imshow(img, 'gray')
    axs[split + 1].imshow(ma.masked_where(lbl == 0, lbl), alpha=0.5)

plt.tight_layout()
plt.show()
