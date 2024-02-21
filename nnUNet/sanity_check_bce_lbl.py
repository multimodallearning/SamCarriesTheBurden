from matplotlib import pyplot as plt
from pathlib import Path
from random import randint
from numpy import ma

img_dir = sorted(Path('/home/ron/Downloads/nnUNet_datasets/imagesTr').glob('*.png'))
lbl_dir = Path('/home/ron/Downloads/nnUNet_datasets/Dataset070_Light_Seg_Graz_bce_all_left/labelsTr')
img_files = sorted(list(img_dir))

idx = randint(0, len(img_files) - 1)
img = plt.imread(img_files[idx])
fig, axs = plt.subplots(1, 2)
axs = axs.flatten()
axs[0].imshow(img, 'gray')

img_name = img_files[idx].stem[:-5] + '.png'
lbl = plt.imread(Path(lbl_dir).joinpath(img_name))
axs[1].imshow(img, 'gray')
axs[1].imshow(lbl, alpha=(lbl > 0).astype(float))

plt.tight_layout()
plt.show()
