import logging
from collections import OrderedDict

import torch
from torch.utils.data import Dataset


class JSRTDataset(Dataset):
    SPLIT_IDX = 160
    CLASS_LABEL = ['right_lung', 'left_lung', 'heart', 'right_clavicle', 'left_clavicle']
    N_CLASSES = len(CLASS_LABEL)
    BCE_POS_WEIGHTS = torch.tensor([4.9105, 6.3263, 10.2737, 106.2339, 106.7618])
    BCE_POS_WEIGHTS = BCE_POS_WEIGHTS.view(N_CLASSES, 1, 1).expand(N_CLASSES, 256, 256)
    # neutral statistics because images are already z-normalized
    IMG_MEAN = 0.
    IMG_STD = 1.

    def __init__(self, mode: str, number_training_samples: int | str = 'all'):
        super().__init__()

        # load data
        data = torch.load('data/jsrt/JSRT_img0_lms.pth', map_location='cpu')
        self.imgs = data['JSRT_img0'].float()  # already z-normalized
        del data
        self.seg_masks = torch.load('data/jsrt/jsrt_seg_masks.pth').float()
        self.data_idx = torch.arange(len(self.imgs))

        # check for equal number of samples
        assert self.imgs.shape[0] == self.seg_masks.shape[0]

        # select images for training or testing
        if mode == 'train':

            if number_training_samples == 'all':
                number_training_samples = self.SPLIT_IDX
            elif number_training_samples > self.SPLIT_IDX:
                raise ValueError(
                    f'number_training_samples {number_training_samples} is greater than the available samples '
                    f'for training {self.SPLIT_IDX}')

            self.imgs = self.imgs[:number_training_samples]
            self.seg_masks = self.seg_masks[:number_training_samples]
            self.data_idx = self.data_idx[:number_training_samples]
        elif mode == 'val':
            self.imgs = self.imgs[self.SPLIT_IDX:][:40]
            self.seg_masks = self.seg_masks[self.SPLIT_IDX:][:40]
            self.data_idx = self.data_idx[self.SPLIT_IDX:][:40]
        elif mode == 'test':
            self.imgs = self.imgs[self.SPLIT_IDX:][40:]
            self.seg_masks = self.seg_masks[self.SPLIT_IDX:][40:]
            self.data_idx = self.data_idx[self.SPLIT_IDX:][40:]
        elif mode == 'all':
            logging.warning('Using all data. Please do not use this mode in context of training.')
        else:
            raise ValueError(f'Unknown mode {mode}')

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        img = self.imgs[idx].unsqueeze(0)  # add channel dimension
        seg_mask = self.seg_masks[idx]
        data_idx = self.data_idx[idx]

        return img, seg_mask, data_idx
