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
theta = torch.eye(2, 3).unsqueeze(0) + torch.randn(1, 2, 3) * .07
affine = affine_grid(theta, x.shape, align_corners=False)
x = grid_sample(x, affine, align_corners=False)
y = grid_sample(y, affine, mode='nearest', align_corners=False)

plt.figure()
x = x.squeeze()
y = y.squeeze()
plt.imshow(x, cmap='gray')
plt.imshow(y.argmax(0), alpha=y.any(0).float() * .5)
plt.title('transformed')

x = x.unsqueeze(0).unsqueeze(0)
y = y.unsqueeze(0)
theta_squared = torch.cat([theta, torch.tensor([0, 0, 1]).view(1, 1, 3).to(theta)], dim=1)
theta_inv = theta_squared.inverse()
theta_inv = theta_inv[:, :2, :]
affine = affine_grid(theta_inv, x.shape, align_corners=False)
x = grid_sample(x, affine, align_corners=False)
y = grid_sample(y, affine, mode='nearest', align_corners=False)

plt.figure()
x = x.squeeze()
y = y.squeeze()
plt.imshow(x, cmap='gray')
plt.imshow(y.argmax(0), alpha=y.any(0).float() * .5)
plt.title('inverse transformed')

plt.show()
