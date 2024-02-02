from pathlib import Path
from typing import List

import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from segment_anything.utils.prompt_utils import PromptExtractor


def sam_prompt_debug_plots(prompt_extractor: PromptExtractor, img: torch.Tensor, initial_seg: torch.Tensor,
                           sam_seg: torch.Tensor, sam_score: torch.Tensor, prompts2use: List[str], save_path: Path):
    assert initial_seg.shape == sam_seg.shape, f"expected initial_seg and sam_seg to have same shape, got {initial_seg.shape} and {sam_seg.shape} respectively"

    img = img.squeeze()
    save_path.mkdir(exist_ok=True)
    for prompt in prompt_extractor.extract():
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        # initial segmentation
        axes[0].imshow(img, 'gray')
        axes[0].imshow(initial_seg[prompt.class_idx], alpha=initial_seg[prompt.class_idx].float())
        axes[0].set_title('initial segmentation')

        # prompt base
        axes[1].imshow(img, 'gray')
        ccl_mask = prompt_extractor.pred_mask[prompt.class_idx]
        axes[1].imshow(ccl_mask, alpha=ccl_mask.float())
        if 'pos_points' in prompts2use:
            axes[1].scatter(prompt.pos_seeds[:, 0], prompt.pos_seeds[:, 1], c='r')
        if 'neg_points' in prompts2use:
            axes[1].scatter(prompt.neg_seeds[:, 0], prompt.neg_seeds[:, 1], c='b')
        if 'box' in prompts2use:
            x_min, y_min, x_max, y_max = prompt.box
            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
            axes[1].add_patch(rect)
        axes[1].set_title('prompt base')

        # sam segmentation
        axes[2].imshow(img, 'gray')
        axes[2].imshow(sam_seg[prompt.class_idx], alpha=sam_seg[prompt.class_idx].float())
        axes[2].set_title(f'sam segmentation\nest DSC: {sam_score[prompt.class_idx]:.4f}')

        # image
        axes[3].imshow(img, 'gray')

        plt.tight_layout()
        plt.savefig(save_path / f'class_{prompt.class_idx}.png', dpi=200)
        plt.close(fig)

def plot_rnd_walk(img:torch.Tensor, initial_masks: torch.Tensor, preprocessed_masks:torch.Tensor,
                  p_hat_rnd_walk: torch.Tensor, save_path: Path):
    save_path.mkdir(exist_ok=True)
    for cls_idx, (init_mask, preproc_mask, p_hat) in enumerate(zip(initial_masks, preprocessed_masks, p_hat_rnd_walk)):
        fig, axs = plt.subplots(1, 4, figsize=(16, 5))
        axs[0].imshow(img, cmap='gray')
        axs[0].imshow(init_mask, alpha=init_mask.float())
        axs[0].set_title('Initial Mask')

        axs[1].imshow(img, cmap='gray')
        axs[1].imshow(preproc_mask, alpha=preproc_mask.float())
        axs[1].set_title('Preprocessed Mask')

        axs[2].imshow(img, cmap='gray')
        axs[2].imshow(p_hat, alpha=(p_hat > 0.5).float())
        axs[2].set_title('Random Walk')

        axs[3].imshow(img, cmap='gray')

        plt.tight_layout()
        plt.savefig(save_path / f'class_{cls_idx}.png', dpi=200)
        plt.close(fig)

