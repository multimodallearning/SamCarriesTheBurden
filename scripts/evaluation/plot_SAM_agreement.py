from matplotlib import pyplot as plt
import h5py
import torch

# load h5 file
f = h5py.File('data/seg_masks/3bdbce9d81b24aa796c389cd2e188313/raw_segmentations_450.h5', 'r')
ds = f['segmentation_mask']
dsc_agreement = [ds[img_name].attrs['SAM_agreement_dsc'] for img_name in ds.keys()]
dsc_agreement = sorted(dsc_agreement)

plt.plot(torch.linspace(0, 1, len(dsc_agreement)), dsc_agreement)
plt.xlabel('Fraction of images')
plt.ylabel('Dice coefficient of SAM agreement')
plt.show()
