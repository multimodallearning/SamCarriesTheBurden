from abc import ABC, abstractmethod

import torch
from torch.nn import functional as F

from segment_anything.sam_mask_decoder_head import SAMMaskDecoderHead
from segment_anything.utils.prompt_utils import PromptExtractor
from utils.segmentation_preprocessing import *


class SegRefiner(ABC):
    @abstractmethod
    def refine(self, seg: torch.Tensor, file_name: str = None) -> torch.Tensor:
        pass


class SegEnhance:
    def __init__(self, refiner: SegRefiner, ccl_selection: str | None, morph_op: str, struct_element: str, radius: int,
                 device: str):
        """
        Takes an initial segmentation mask and refines it with given segmentation refiner.
        Args:
            refiner: segmentation refiner
            ccl_selection: selection of connected component. Valid values: 'largest', 'highest_probability'. Choose None for connected component analysis.
            morph_op: choose morphological operation. Valid values: 'erosion', 'dilation'
            struct_element: structuring element for morphological operation. Valid values: 'square', 'disk', 'diamond', 'star'
            radius: radius of the structuring element. Choose 0 for identity mapping.
            device: pytorch device to use for computation
        """

        self.last_preprocessed_seg = None
        self.refiner = refiner
        self.num_iter = None  # will be calculated on the fly
        if ccl_selection is None:
            self.ccl = lambda mask: mask
        else:
            self.ccl = lambda mask: remove_all_but_one_connected_component(mask, ccl_selection, num_iter=self.num_iter)

        # morphological operation
        struct = {
            'square': square,
            'disk': disk,
            'diamond': diamond,
            'star': star
        }[struct_element]
        if struct == square and radius == 0:
            # handling identity for square structuring element
            radius = 1
        kernel = torch.from_numpy(struct(radius, dtype=int)).float().to(device)
        morph_op = {
            'erosion': erosion,
            'dilation': dilation
        }[morph_op]
        if radius == 0 or (struct == square and radius == 1):
            # identity
            self.morph_op = lambda mask: mask
        else:
            self.morph_op = lambda mask: morph_op(mask, kernel, engine='convolution')

    @torch.inference_mode()
    def enhance(self, seg: torch.Tensor, file_name: str = None) -> torch.Tensor:
        assert seg.ndim == 3, "seg should be 3D tensor of shape (C, H, W)"
        self.num_iter = max(seg.shape[-2:])  # give connected component analysis space to spread its wings

        seg = self.ccl(seg)  # HPO found no benefit from morphological operation before connected component labeling
        self.last_preprocessed_seg = self.morph_op(seg.unsqueeze(0).float()).squeeze(0)
        result = self.refiner.refine(seg, file_name)

        return result  # could also return a tuple of seg and est_dice


class SAMSegRefiner(SegRefiner):
    def __init__(self, sam_type: str, device: str, prompts2use: list[list[str]] | list[str]):
        if sam_type == 'SAM':
            sam_checkpoint = "data/sam_vit_h_4b8939.pth"
            sam_model_type = "vit_h"
            img_embedding_h5 = "data/graz_sam_img_embedding.h5"
        elif sam_type == 'MedSAM':
            sam_checkpoint = "data/medsam_vit_b.pth"
            sam_model_type = "vit_b"
            img_embedding_h5 = "data/graz_medsam_img_embedding.h5"
        else:
            raise NotImplementedError(f'Unknown SAM type: {sam_type}')
        self.sam_predictor = SAMMaskDecoderHead(sam_checkpoint, sam_model_type, device, img_embedding_h5)
        # check if prompts2use is nested list
        if isinstance(prompts2use[0], list):
            self.prompts2use1st = prompts2use[0]
            self.prompts2use2nd = prompts2use[1]
            self.self_refine = True
        else:
            self.prompts2use1st = prompts2use
            self.prompts2use2nd = None
            self.self_refine = False

    def refine(self, seg: torch.Tensor, file_name: str) -> torch.Tensor:
        seg = seg.bool()
        prompt_extractor = PromptExtractor(seg)
        prompts = prompt_extractor.extract()

        est_dice = torch.full((seg.shape[0],), float('nan'))
        for prompt in prompts:
            mask, mask_score, mask_prev_iter = self.sam_predictor.predict_mask(file_name, prompt, self.prompts2use1st)
            if self.prompts2use2nd is not None:
                mask, mask_score, _ = self.sam_predictor.predict_mask(file_name, prompt, self.prompts2use2nd,
                                                                      mask_prev_iter)

            mask = F.interpolate(mask.float(), size=seg.shape[-2:], mode='nearest-exact')
            seg[prompt.class_idx] = mask.squeeze()
            # convert Jaccard to Dice
            est_dice[prompt.class_idx] = 2 * mask_score / (1 + mask_score)

        return seg, est_dice


if __name__ == '__main__':
    refiner = SAMSegRefiner('SAM', 'cpu', ['pos_points', 'neg_points', 'box'])
    seg_enhancer = SegEnhance(refiner, 'highest_probability', 'dilation', 'disk', 3, 'cpu')
    seg = torch.rand(16, 384, 224)
    seg = seg_enhancer.enhance(seg, "0003_0662359226_01_WRI-R1_M011")
