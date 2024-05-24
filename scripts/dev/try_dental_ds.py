import cv2
from matplotlib import pyplot as plt
from scripts.dental_dataset import DentalDataset

img = cv2.imread('/home/ron/Documents/SemiSAM/data/DentalSeg/img/102.jpg', cv2.IMREAD_GRAYSCALE)
seg_mask = cv2.imread('/home/ron/Documents/SemiSAM/data/DentalSeg/masks_machine/102.png', cv2.IMREAD_GRAYSCALE)

fig, axs = plt.subplots(2, 1)
axs[0].imshow(img, cmap='gray')

axs[1].imshow(img, cmap='gray')
axs[1].imshow(seg_mask, alpha=seg_mask.astype(bool).astype(float), interpolation='nearest', cmap='tab20')

# resize
img = cv2.resize(img, (384, 224))
seg_mask = cv2.resize(seg_mask, (384, 224), interpolation=cv2.INTER_NEAREST)

fig, axs = plt.subplots(2, 1)
axs[0].imshow(img, cmap='gray')

axs[1].imshow(img, cmap='gray')
axs[1].imshow(seg_mask, alpha=seg_mask.astype(bool).astype(float), interpolation='nearest', cmap='jet')

ds = DentalDataset('train', 10)
img, mask, filestem = ds[4]
img = img.squeeze()

fig, axs = plt.subplots(2, 1)
axs[0].imshow(img, cmap='gray')

axs[1].imshow(img, cmap='gray')
axs[1].imshow(mask.float().argmax(0), alpha=mask.any(0).float(), interpolation='nearest', cmap='jet')

plt.show()