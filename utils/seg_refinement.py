from abc import ABC, abstractmethod
from pathlib import Path

import cv2
from torch.nn import functional as F

from segment_anything.sam_mask_decoder_head import SAMMaskDecoderHead
from segment_anything.utils.prompt_utils import PromptExtractor
from utils import segmentation_preprocessing
from utils.random_walk import sparse_cols, sparse_rows, sparseMultiGrid
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
            img_embedding_h5 = "data/dental_sam_img_embedding.h5"
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
            assert len(prompts2use[1]) > 0, "2nd prompt list should not be empty"
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


class RndWalkSegRefiner(SegRefiner):
    def __init__(self, background_erosion_radius: int, laplace_sigma: float, laplace_lambda: float = 1):
        self.background_erosion_radius = background_erosion_radius
        self.laplace_lambda = laplace_lambda
        self.laplace_sigma = laplace_sigma
        self.last_input_seg = None  # buffer for plotting

        self.img_path = Path('data/DentalSeg/img')

    def refine(self, seg: torch.Tensor, file_name: str) -> torch.Tensor:
        device = seg.device
        self.last_input_seg = seg
        img = cv2.imread(str(self.img_path / (file_name + '.jpg')), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (seg.shape[-1], seg.shape[-2]))
        img = torch.from_numpy(img).to(device)

        # sanity checks
        assert img.ndim == 2, 'img should be 2D'
        assert img.dtype == torch.uint8, 'img should be in range [0, 255]'
        H, W = img.shape
        assert seg.ndim == 3
        assert seg.shape[1] == H and seg.shape[2] == W

        # add a background class to the initial segmentation
        background = torch.logical_not(seg.any(0))
        if self.background_erosion_radius > 1:
            background = segmentation_preprocessing.erode_mask_with_disc_struct(background.unsqueeze(0),
                                                                                radius=self.background_erosion_radius)
        initial_segmentation = torch.cat([background.unsqueeze(0), seg], dim=0)

        linear_idx = torch.arange(H * W, device=device).view(H, W)
        idx_mask = initial_segmentation.any(0)
        seeded = linear_idx[idx_mask]
        unseeded = linear_idx[~idx_mask]

        L = self.laplace_matrix(img.float())
        L_u = sparse_rows(sparse_cols(L, unseeded), unseeded)
        B = sparse_rows(sparse_cols(L, unseeded), seeded)

        u_s = initial_segmentation[:, idx_mask].t()

        b = torch.mm(-B.t(), u_s.float())
        u_u = sparseMultiGrid(L_u, b)

        # combine prediction (u_u) with the labels for the known pixels (u_s)
        p_hat = torch.zeros(H * W, u_s.shape[-1], device=device)
        p_hat[seeded] = u_s.float()
        p_hat[unseeded] = u_u

        p_hat = p_hat.view(H, W, -1).permute(2, 0, 1)
        # remove the background class
        p_hat = p_hat[1:]
        y_hat = p_hat > 0.5

        return y_hat, None  # dummy to match the signature of SAMSegRefiner

    def laplace_matrix(self, img):
        device = img.device
        if not img.dtype is torch.float:
            raise TypeError('an image of type float is expected.')
        H, W = img.size()
        # create 1D index vector
        ind = torch.arange(H * W, device=device).view(H, W)
        # select left->right neighbours
        ii = torch.cat((ind[:, 1:].reshape(-1, 1), ind[:, :-1].reshape(-1, 1)), 1)
        val = torch.exp(-(img.take(ii[:, 0]) - img.take(ii[:, 1])) ** 2 / self.laplace_sigma ** 2)

        # create first part of neigbourhood matrix (similar to setFromTriplets in Eigen)
        A = torch.sparse.FloatTensor(ii.t(), val, torch.Size([H * W, H * W]))
        # select up->down neighbours
        ii = torch.cat((ind[1:, :].reshape(-1, 1), ind[:-1, :].reshape(-1, 1)), 1)
        val = torch.exp(-(img.take(ii[:, 0]) - img.take(ii[:, 1])) ** 2 / self.laplace_sigma ** 2)

        # create second part of neigbourhood matrix (similar to setFromTriplets in Eigen)
        A = A + torch.sparse.FloatTensor(ii.t(), val, torch.Size([H * W, H * W]))
        # make symmetric (add down->up and right->left)
        A = A + A.t()
        # compute degree matrix (diagonal sum)
        D = torch.sparse.sum(A, 0).to_dense()
        # put D and A together
        val = .00001 + self.laplace_lambda * D
        L = torch.sparse.FloatTensor(torch.cat((ind.view(1, -1), ind.view(1, -1)), 0), val,
                                     torch.Size([H * W, H * W]))
        L += (A * (-self.laplace_lambda))
        return L


if __name__ == '__main__':
    refiner = SAMSegRefiner('SAM', 'cpu', ['pos_points', 'neg_points', 'box'])
    seg_enhancer = SegEnhance(refiner, 'highest_probability', 'dilation', 'disk', 3, 'cpu')
    seg = torch.rand(16, 384, 224)
    seg = seg_enhancer.enhance(seg, "0003_0662359226_01_WRI-R1_M011")
