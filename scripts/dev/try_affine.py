from scripts.seg_grazpedwri_dataset import LightSegGrazPedWriDataset
import torch
from torch.nn.functional import affine_grid, grid_sample
from matplotlib import pyplot as plt

ds = LightSegGrazPedWriDataset('val')

x, y, _ = ds[1]
plt.figure()
plt.imshow(x[0], cmap='gray')
plt.imshow(y.argmax(0), alpha=y.any(0).float() * .5)
plt.title('original')

x = x.unsqueeze(0)
y = y.unsqueeze(0)
theta = torch.eye(2, 3).unsqueeze(0) + torch.randn(1, 2, 3) * .1
affine = affine_grid(theta, x.shape, align_corners=False)
x = grid_sample(x, affine, align_corners=False)
y = grid_sample(y, affine, mode='nearest', align_corners=False)

plt.figure()
x = x.squeeze()
y = y.squeeze()
plt.imshow(x, cmap='gray')
plt.imshow(y.argmax(0), alpha=y.any(0).float() * .5)

plt.show()
