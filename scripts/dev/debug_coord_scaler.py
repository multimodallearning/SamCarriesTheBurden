from matplotlib import pyplot as plt
from segment_anything.utils.prompt_utils import scale_box, scale_coords
import torch
import cv2


def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


img = cv2.imread('data/img_only_front_all_left/2639_0713211442_01_WRI-R1_M005.png', cv2.IMREAD_GRAYSCALE)
original_img = cv2.resize(img, (224, 384))

original_box = torch.tensor([49, 176, 104, 382])
original_coords = torch.tensor(
    [[70, 277], [184, 100], [37, 38], [147, 164], [129, 120], [100, 116], [116, 150], [157, 108], [89, 138], [195, 78],
     [158, 52], [119, 50], [89, 53], [64, 73], [29, 15], [135, 266]])

plt.figure(figsize=(10, 10))
plt.imshow(original_img, 'gray')
show_box(original_box, plt.gca())
show_points(original_coords, torch.ones(len(original_coords)), plt.gca())
plt.title('Original image')

original_size = original_img.shape[:2]
target_size = img.shape[:2]

scaled_coords = scale_coords(original_coords, original_size, target_size)
scaled_box = scale_box(original_box.unsqueeze(0), original_size, target_size)

plt.figure(figsize=(10, 10))
plt.imshow(img, 'gray')
show_box(scaled_box[0], plt.gca())
show_points(scaled_coords, torch.ones(len(scaled_coords)), plt.gca())
plt.title('Scaled image')

plt.show()
