from pathlib import Path

import albumentations as A
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from scripts.cvat_parser import CVATParser


class SegGrazPedWriDataset(Dataset):
    # calculated over training split
    IMG_MEAN = 0.3505533917353781
    IMG_STD = 0.22763733675869177

    # bone label
    BONE_LABEL = sorted([
        'Radius',
        'Ulna',
        'Os scaphoideum',
        'Os lunatum',
        'Os triquetrum',
        'Os pisiforme',
        'Os trapezium',
        'Os trapezoideum',
        'Os capitatum',
        'Os hamatum',
        'Ossa metacarpalia I',
        'Ossa metacarpalia II',
        'Ossa metacarpalia III',
        'Ossa metacarpalia IV',
        'Ossa metacarpalia V',
        'Epiphyse Radius',
        'Epiphyse Ulna',
        'Phalanx proximalis I',
        'Phalanx proximalis II',
        'Phalanx proximalis III',
        'Phalanx proximalis IV',
        'Phalanx proximalis V',
        'Phalanx media II',
        'Phalanx media III',
        'Phalanx media IV',
        'Phalanx media V',
        'Phalanx distalis I',
        'Phalanx distalis II',
        'Phalanx distalis III',
        'Phalanx distalis IV',
        'Phalanx distalis V',
        'E Ossa metacarpalia I',
        'E Ossa metacarpalia II',
        'E Ossa metacarpalia III',
        'E Ossa metacarpalia IV',
        'E Ossa metacarpalia V',
        'E Phalanx proximalis I',
        'E Phalanx proximalis II',
        'E Phalanx proximalis III',
        'E Phalanx proximalis IV',
        'E Phalanx proximalis V',
        'E Phalanx media II',
        'E Phalanx media III',
        'E Phalanx media IV',
        'E Phalanx media V',
        'E Phalanx distalis I',
        'E Phalanx distalis II',
        'E Phalanx distalis III',
        'E Phalanx distalis IV',
        'E Phalanx distalis V'])
    BONE_LABEL_MAPPING = {k: v for k, v in zip(BONE_LABEL, range(len(BONE_LABEL)))}
    N_CLASSES = len(BONE_LABEL)
    # CE weights including background
    SQRT_INV_CLASS_FREQ = torch.tensor([1.2008, 26.7079, 36.0101, 43.8595, 44.6148, 36.7601, 118.9976,
                                        260.2164, 237.5440, 131.9195, 287.6801, 210.5602, 199.0454, 313.0576,
                                        160.8181, 70.1868, 56.1553, 57.5872, 60.1601, 59.5554, 10.4827,
                                        18.7768, 8.4333, 9.8873, 13.0366, 19.1863, 11.5521, 13.3597,
                                        15.6073, 13.1041, 7.8459, 6.9267, 7.2839, 7.7481, 7.3461,
                                        66.6644, 115.3286, 104.9682, 103.5980, 130.4049, 71.1340, 60.3362,
                                        56.8825, 99.9221, 29.2800, 27.5295, 32.7088, 24.5364, 22.2836,
                                        3.5138, 4.3671])
    # BCE weights not including background
    POS_CLASS_WEIGHT = torch.tensor([7.0743e+02, 1.2868e+03, 1.9095e+03, 1.9759e+03, 1.3410e+03, 1.4062e+04,
                                     6.7248e+04, 5.6040e+04, 1.7283e+04, 8.2192e+04, 4.4031e+04, 3.9347e+04,
                                     9.7333e+04, 2.5684e+04, 4.8914e+03, 3.1308e+03, 3.2926e+03, 3.5934e+03,
                                     3.5216e+03, 1.0813e+02, 3.4916e+02, 6.9634e+01, 9.6089e+01, 1.6779e+02,
                                     3.6459e+02, 1.3154e+02, 1.7626e+02, 2.4092e+02, 1.6954e+02, 6.0136e+01,
                                     4.6651e+01, 5.1692e+01, 5.8622e+01, 5.2596e+01, 4.4127e+03, 1.3209e+04,
                                     1.0942e+04, 1.0658e+04, 1.6888e+04, 5.0244e+03, 3.6145e+03, 3.2125e+03,
                                     9.9151e+03, 8.5045e+02, 7.5168e+02, 1.0615e+03, 5.9691e+02, 4.9216e+02,
                                     1.1262e+01, 1.7941e+01])

    def __init__(self, mode: str, rescale_HW: tuple = (384, 224), lbl_idx_to_use: list = None):
        """
        :param mode: data split mode [training, validation, testing]
        :param rescale_HW: rescale image and ground truth (should be close to ratio of 1.75 as possible). None will not rescale.
        :param lbl_idx_to_use: list of indices of labels to use. None will use all labels.
        """
        super().__init__()
        # load data meta and other information
        self.df_meta = pd.read_csv('data/dataset.csv', index_col='filestem')
        assert mode in ['train', 'test'], f'Unknown mode {mode}'
        # init ground truth parser considering the data split
        xml_files = list(Path('data/cvat_annotation_xml').glob(f'annotations_{mode}[1-9].xml'))
        self.gt_parser = CVATParser(xml_files, True, False, True)
        # drop labels not in lbl_idx_to_use
        if lbl_idx_to_use:
            self.BONE_LABEL = [self.BONE_LABEL[i] for i in lbl_idx_to_use]
            self.BONE_LABEL_MAPPING = {k: v for k, v in zip(self.BONE_LABEL, range(len(self.BONE_LABEL)))}
            self.N_CLASSES = len(self.BONE_LABEL)
            self.POS_CLASS_WEIGHT = self.POS_CLASS_WEIGHT[lbl_idx_to_use]
            # handling background class
            self.SQRT_INV_CLASS_FREQ = torch.cat([
                self.SQRT_INV_CLASS_FREQ[0].unsqueeze(0),  # background class
                self.SQRT_INV_CLASS_FREQ[1:][lbl_idx_to_use]
            ])
            assert self.N_CLASSES == len(self.BONE_LABEL_MAPPING) == len(self.SQRT_INV_CLASS_FREQ) - 1 == len(
                self.POS_CLASS_WEIGHT) == len(lbl_idx_to_use)

        # get file names
        # filter files to front view only
        projection_mask = self.df_meta['projection'] == 1
        files_with_annotations_mask = self.df_meta.index.isin(self.gt_parser.available_file_names)
        self.available_file_names = self.df_meta[projection_mask & files_with_annotations_mask].index.tolist()

        # init static transformer
        additional_targets = {lbl: 'mask' for lbl in self.BONE_LABEL}
        self.rescale = A.Resize(height=rescale_HW[0], width=rescale_HW[1], interpolation=cv2.INTER_LINEAR,
                                always_apply=True)
        self.rescale = A.Compose([self.rescale], additional_targets=additional_targets)

        # load img into memory
        img_path = Path('data/img_only_front_all_left')
        self.data = dict()
        for file_name in tqdm(self.available_file_names, unit='img', desc=f'Loading data for {mode}'):
            data_dict = dict()
            data_dict['image'] = cv2.imread(str(img_path.joinpath(file_name).with_suffix('.png')), cv2.IMREAD_GRAYSCALE)

            seg_masks = self.gt_parser.extract_masks(file_name)
            seg_masks = CVATParser.cvt_mask_list_2_dict(seg_masks)
            data_dict.update(seg_masks)

            need2flip = self.df_meta.loc[file_name, 'laterality'] == 'R'  # flip to left hand

            # rescale
            data_dict = A.Compose([self.rescale], additional_targets=additional_targets)(**data_dict)

            # stack masks
            for lbl in self.BONE_LABEL:
                try:
                    data_dict[lbl] = torch.from_numpy(data_dict[lbl])
                except KeyError:
                    data_dict[lbl] = torch.zeros(data_dict['image'].shape)  # add empty mask if not annotated
            y = torch.stack([data_dict[lbl] for lbl in self.BONE_LABEL], dim=0)
            if need2flip:  # image is already stored as flipped image
                y = torch.flip(y, dims=[-1])

            # numpy image to tensor and add channel dimension
            img = torch.from_numpy(data_dict['image']).unsqueeze(0)
            img = img / 255

            self.data[file_name] = {'image': img, 'mask': y.float()}

    def __len__(self):
        return len(self.available_file_names)

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor, str):
        """
        get item by index
        :param index: index of item
        :return: image, ground truth, file name
        """
        file_name = self.available_file_names[index]
        data_dict = self.data[file_name]
        x, y = data_dict['image'], data_dict['mask']

        return x, y.float(), file_name


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from numpy import ma
    from random import randint

    lbl_idx_independent_set = [0, 2, 3, 6, 7, 8, 9, 10, 12, 16, 17, 23, 24, 30, 33, 34, 35, 36, 38, 40, 42, 43, 44, 47,
                               49]
    ds = SegGrazPedWriDataset('train', lbl_idx_to_use=None)
    idx = 1  # randint(0, len(ds) - 1)
    x, y, filename = ds[idx]
    plt.imshow(x.squeeze(0), cmap='gray')
    for lbl, mask in zip(ds.BONE_LABEL, y):
        if mask.sum() == 0:
            continue
        plt.figure()
        plt.imshow(x.squeeze(0), cmap='gray')
        plt.imshow(ma.masked_where(mask == 0, mask), alpha=0.5)
        plt.title(lbl)
    plt.show()
