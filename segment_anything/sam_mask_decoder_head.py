from typing import List, Tuple

from segment_anything import sam_model_registry
from pathlib import Path
import torch
import h5py
from segment_anything.utils.prompt_utils import Prompt, scale_box, scale_coords
from torch.nn import functional as F
from copy import deepcopy


class SAMMaskDecoderHead:
    def __init__(self, sam_checkpoint: str | Path, model_type: str, device: str, img_embedding_h5: str | Path):
        sam_checkpoint = Path(sam_checkpoint)
        img_embedding_h5 = Path(img_embedding_h5)
        self.device = torch.device(device)

        h5_file = h5py.File(img_embedding_h5, 'r')
        self.img_embedding = h5_file['img_embedding']
        self.img_enc_img_size = int(h5_file.attrs['img_encoder_img_size'])

        assert h5_file.attrs['checkpoint'] == sam_checkpoint.name, 'SAM checkpoint mismatch'
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.prompt_encoder = sam.prompt_encoder.to(device=self.device, non_blocking=True)
        self.mask_decoder = sam.mask_decoder.to(device=self.device, non_blocking=True)
        self.mask_threshold = sam.mask_threshold
        del sam

    @torch.inference_mode()
    def predict_mask(self, img_name: str, given_prompt: Prompt, prompt2use: str | List[str],
                     mask_prev_iter: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        known_prompts = ['pos_points', 'neg_points', 'box']
        if isinstance(prompt2use, str):
            prompt2use = [prompt2use]
        assert all([p in known_prompts for p in prompt2use]), f'Prompt must be one of {known_prompts}'
        prompt = deepcopy(given_prompt)

        if 'pos_points' in prompt2use:
            assert prompt.pos_seeds is not None, 'pos_seeds are not available'
            prompt.pos_seeds = scale_coords(prompt.pos_seeds, prompt.img_size,
                                            self.img_embedding[img_name].attrs['input_size'])
        else:
            prompt.pos_seeds = None

        if 'neg_points' in prompt2use:
            assert prompt.neg_seeds is not None, 'neg_seeds are not available'
            prompt.neg_seeds = scale_coords(prompt.neg_seeds, prompt.img_size,
                                            self.img_embedding[img_name].attrs['input_size'])
        else:
            prompt.neg_seeds = None

        if 'box' in prompt2use:
            assert prompt.box is not None, 'box is not available'
            prompt.box = prompt.box.unsqueeze(0)
            prompt.box = scale_box(prompt.box, prompt.img_size, self.img_embedding[img_name].attrs['input_size'])
            prompt.box = prompt.box.float().to(self.device, non_blocking=True)
        else:
            prompt.box = None

        input_points = []
        input_labels = []
        if prompt.pos_seeds is not None:
            input_points.append(prompt.pos_seeds)
            input_labels.append(torch.ones(prompt.pos_seeds.shape[0]))
        if prompt.neg_seeds is not None:
            input_points.append(prompt.neg_seeds)
            input_labels.append(torch.zeros(prompt.neg_seeds.shape[0]))
        input_points = torch.cat(input_points).float() if len(input_points) > 0 else None
        input_labels = torch.cat(input_labels).int() if len(input_labels) > 0 else None

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(input_points.unsqueeze(0), input_labels.unsqueeze(0)) if input_points is not None else None,
            boxes=prompt.box,
            masks=mask_prev_iter,
        )

        ds_img_embedding = self.img_embedding[img_name]
        features = ds_img_embedding['features'][:]
        features = torch.from_numpy(features).to(device=self.device, non_blocking=True)

        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks,
                                       ds_img_embedding.attrs['input_size'].tolist(),
                                       ds_img_embedding.attrs['original_size'].tolist())
        masks = masks > self.mask_threshold

        return masks, iou_predictions, low_res_masks

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.img_enc_img_size, self.img_enc_img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks
