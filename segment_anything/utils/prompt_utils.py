from dataclasses import dataclass
from functools import cached_property
from typing import Tuple

import torch
from torch.nn import functional as F

from segment_anything.utils.transforms import ResizeLongestSide


@dataclass
class Prompt:
    class_idx: int
    img_size: Tuple[int, int]  # (H, W)
    pos_seeds: torch.Tensor = None
    neg_seeds: torch.Tensor = None
    box: torch.Tensor = None
    mask_logits: torch.Tensor = None


class PromptExtractor:
    def __init__(self, pred_mask: torch.Tensor):
        """
        Extracts prompts from a predicted mask.
        Args:
            pred_mask: 3D tensor of shape (C, H, W) with boolean values
        """
        assert pred_mask.ndim == 3, "pred_mask should be 3D tensor of shape (C, H, W)"
        assert pred_mask.dtype == torch.bool, "pred_mask should be boolean tensor"

        self.pred_mask = pred_mask
        self.num_classes = pred_mask.shape[0]

    def _extract_seeds(self, class_idx: int) -> torch.Tensor:
        assert class_idx < self.num_classes, "class_idx exceeds number of classes"
        class_mask = self.pred_mask[class_idx] & self.masked_non_overlapping_label_areas
        num_seeds = class_mask.sum()
        if num_seeds == 0:
            return None
        else:
            coords = torch.nonzero(class_mask, as_tuple=False)
            coords = coords.float().mean(0, keepdim=True).round().int()
            return coords.flip(-1)  # HW -> WH

    def _extract_box(self, class_idx: int) -> torch.Tensor:
        assert class_idx < self.num_classes, "class_idx exceeds number of classes"
        class_mask = self.pred_mask[class_idx]
        if class_mask.sum() == 0:
            return None
        else:
            coords = torch.nonzero(class_mask, as_tuple=False)
            coords = coords.float()
            x_min = coords[:, 1].min()
            x_max = coords[:, 1].max()
            y_min = coords[:, 0].min()
            y_max = coords[:, 0].max()

            box = torch.tensor([x_min, y_min, x_max, y_max], device=self.pred_mask.device).round().int()
            return box

    @cached_property
    def seeds(self):
        return [self._extract_seeds(i) for i in range(self.num_classes)]

    @cached_property
    def masked_non_overlapping_label_areas(self):
        return self.pred_mask.sum(0) < 2

    # adapted from https://github.com/computational-cell-analytics/micro-sam/blob/83997ff4a471cd2159fda4e26d1445f3be79eb08/micro_sam/prompt_based_segmentation.py#L71
    def _compute_logits_from_mask(self, class_idx, eps=1e-3):

        def inv_sigmoid(x):
            return torch.log(x / (1 - x))

        class_mask = self.pred_mask[class_idx]

        logits = torch.zeros_like(class_mask, dtype=torch.float)
        logits[class_mask] = 1 - eps
        logits[class_mask.logical_not()] = eps
        logits = inv_sigmoid(logits)

        # resize to the expected mask shape of SAM (256x256)
        assert logits.ndim == 2
        expected_shape = (256, 256)

        if logits.shape == expected_shape:  # shape matches, do nothing
            pass

        elif logits.shape[0] == logits.shape[1]:  # shape is square
            trafo = ResizeLongestSide(expected_shape[0])
            logits = logits.unsqueeze(0).unsqueeze(0)
            logits = trafo.apply_image_torch(logits).squeeze()  # todo: doc says could not exactly match apply_image

        else:  # shape is not square
            # resize the longest side to expected shape
            trafo = ResizeLongestSide(expected_shape[0])
            logits = logits.unsqueeze(0).unsqueeze(0)
            logits = trafo.apply_image_torch(logits).squeeze()  # todo: doc says could not exactly match apply_image

            # pad the other side
            h, w = logits.shape
            padh = expected_shape[0] - h
            padw = expected_shape[1] - w
            # IMPORTANT: need to pad with zero, otherwise SAM doesn't understand the padding
            pad_width = (0, padw, 0, padh)
            logits = F.pad(logits, pad_width, mode="constant", value=0)  # todo: could be wrong translated from numpy

        logits = logits.unsqueeze(0)
        assert logits.shape == (1, 256, 256), f'Expected logits to have shape (1, 256, 256), got {logits.shape}'
        return logits

    def extract(self, seeds: bool = True, boxes: bool = True, mask: bool = False) -> list[Prompt]:
        """
        Extracts prompts from the predicted mask.
        Args:
            seeds: whether to extract seeds
            boxes: whether to extract boxes
            mask: whether to extract mask logits (not working yet)
        Returns:
            list of prompts with all coordinates in (x, y) format
        """
        prompts = []
        for class_idx in range(self.num_classes):
            # Skip classes with no initial segmentation
            if self.seeds[class_idx] is None:
                continue

            p = Prompt(class_idx, self.pred_mask.shape[-2:])

            if seeds:
                p.pos_seeds = self.seeds[class_idx]
                p.neg_seeds = torch.cat(
                    [self.seeds[i] for i in range(self.num_classes) if i != class_idx and self.seeds[i] is not None])

            if boxes:
                p.box = self._extract_box(class_idx)

            if mask:
                # todo: not working yet
                p.mask_logits = self._compute_logits_from_mask(class_idx)

            prompts.append(p)
        return prompts

def scale_coords(coords: torch.Tensor, original_size: Tuple[int, ...], target_size: Tuple[int, ...]) -> torch.Tensor:
    """
    Scales coordinates from original_size to target_size.
    Args:
        coords: tensor of shape (N, 2) with coordinates in (x, y) format
        original_size: tuple of length (H, W) with original image size
        target_size: tuple of length (H, W) with target image size
    Returns:
        tensor of shape (N, 2) with scaled coordinates
    """
    assert coords.ndim == 2, "coords should be 2D tensor of shape (N, 2)"
    assert coords.shape[1] == len(original_size) == len(target_size), "coords should have same number of dimensions as original_size and target_size"

    original_size = torch.tensor(original_size, dtype=torch.float, device=coords.device)
    target_size = torch.tensor(target_size, dtype=torch.float, device=coords.device)

    coords = coords.float()
    coords = coords * (target_size / original_size).flip(-1) # because SAM takes points in (x, y) format, but shape is (H, W)
    return coords

def scale_box(box: torch.Tensor, original_size: Tuple[int, ...], target_size: Tuple[int, ...]) -> torch.Tensor:
    """
    Scales box from original_size to target_size.
    Args:
        box: tensor of shape (N, 4) with box coordinates in (x_min, y_min, x_max, y_max) format
        original_size: tuple of length (H, W) with original image size
        target_size: tuple of length (H, W) with target image size
    Returns:
        tensor of shape (N, 4) with box coordinates in (x_min, y_min, x_max, y_max) format
    """
    assert box.ndim == 2, "box should be 2D tensor of shape (N, 4)"
    assert box.shape[1] == 4, "box should have length 4"

    box_as_coords = box.reshape(-1, 2)
    box_as_coords = scale_coords(box_as_coords, original_size, target_size)
    return box_as_coords.reshape(-1, 4)