import json
import logging
from pathlib import Path
from random import randint

import albumentations as A
import cv2
import h5py
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.cvat_parser import CVATParser


class LightSegGrazPedWriDataset(Dataset):
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
        'Epiphyse Ulna'])
    BONE_LABEL_MAPPING = {k: v for k, v in zip(BONE_LABEL, range(len(BONE_LABEL)))}
    N_CLASSES = len(BONE_LABEL)
    # BCE weights not including background
    POS_CLASS_WEIGHT = torch.tensor([108.1348, 349.1551, 69.6342, 96.0886, 167.7897, 364.5914, 131.5362,
                                     176.2591, 240.9182, 169.5408, 60.1363, 46.6512, 51.6916, 58.6216,
                                     52.5956, 11.2623, 17.9409])

    def __init__(self, mode: str, number_training_samples: int | str = 'all', rescale_HW: tuple = (384, 224)):
        """
        :param mode: data split mode [training, validation, testing]
        :param number_training_samples: number of training samples to use. If 'all', use all available samples.
        :param rescale_HW: rescale image and ground truth (should be close to ratio of 1.75 as possible). None will not rescale.
        :param lbl_idx_to_use: list of indices of labels to use. None will use all labels.
        """
        super().__init__()
        # load data meta and other information
        self.df_meta = pd.read_csv('data/dataset.csv', index_col='filestem')
        # init ground truth parser considering the data split
        if mode == 'train':
            xml_files = list(Path('data/cvat_annotation_xml').glob(f'annotations_train[1-9].xml'))
        elif mode in ['val', 'test']:
            xml_files = [Path(f'data/cvat_annotation_xml/annotations_{mode}.xml')]
        else:
            raise ValueError(f'Unknown mode {mode}')
        self.gt_parser = CVATParser(xml_files, True, False, True)

        # get file names
        # filter files to front view only
        projection_mask = self.df_meta['projection'] == 1
        files_with_annotations_mask = self.df_meta.index.isin(self.gt_parser.available_file_names)
        self.available_file_names = self.df_meta[projection_mask & files_with_annotations_mask].index.tolist()

        # get subset of training samples
        if mode == 'train' and number_training_samples != 'all':
            # read as series
            training_files = pd.read_csv('data/successively_training_files_order.csv')['file_stem']
            assert len(training_files) == len(self.available_file_names), 'files are missing or duplicated'
            assert number_training_samples <= len(training_files), 'number_training_samples is larger than available files'
            selected_files = training_files[:number_training_samples]
            self.available_file_names = selected_files.tolist()
        elif mode != 'train' and number_training_samples != 'all':
            logging.warning(f'number_training_samples is not used for mode {mode}')

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


class SavedSegGrazPedWriDataset(Dataset):
    # calculated over training split
    IMG_MEAN = LightSegGrazPedWriDataset.IMG_MEAN
    IMG_STD = LightSegGrazPedWriDataset.IMG_STD
    # copy all attributes from LightSegGrazPedWriDataset
    BONE_LABEL = LightSegGrazPedWriDataset.BONE_LABEL
    BONE_LABEL_MAPPING = {k: v for k, v in zip(BONE_LABEL, range(len(BONE_LABEL)))}
    N_CLASSES = len(BONE_LABEL)

    def __init__(self, saved_seg_path: str, use_500_split: bool, rescale_HW: tuple = (384, 224)):
        """
        Dataset that loads images with their stored segmentation. Not include images from training or testing split.
        :param saved_seg_path: path to saved segmentation. Has to be h5 file.
        :param use_500_split: if True, only use the 500 split of once randomly sampled unlabeled data.
        :param rescale_HW: rescale image and ground truth (should be close to ratio of 1.75 as possible). None will not rescale.
        """

        super().__init__()
        h5_file = h5py.File(saved_seg_path, 'r')
        lbl_loaded = json.loads(h5_file.attrs['labels'])
        assert lbl_loaded == self.BONE_LABEL_MAPPING, 'Loaded labels do not match'

        # load data meta and other information
        self.img_path = Path('data/img_only_front_all_left')
        self.ds_saved_seg = h5_file['segmentation_mask']

        # get file names
        if use_500_split:
            self.available_file_names = pd.read_csv('data/500unlabeled_sample.csv')['filestem'].tolist()
        else:
            logging.warning('Using all available files in saved segmentations!')
            self.available_file_names = list(self.ds_saved_seg.keys())

        # init transformation
        self.resize_lbl = lambda x: F.interpolate(x.float().unsqueeze(0), size=rescale_HW, mode='nearest').squeeze(0)

    def __len__(self):
        return len(self.available_file_names)

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor, str):
        """
        get item by index
        :param index: index of item
        :return: image, ground truth, file name
        """
        file_name = self.available_file_names[index]

        # segmentation mask
        seg_masks = torch.from_numpy(self.ds_saved_seg[file_name][:])
        y = self.resize_lbl(seg_masks)

        # numpy image to tensor and add channel dimension
        img = cv2.imread(str(self.img_path.joinpath(file_name).with_suffix('.png')), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, y.shape[-2:][::-1], interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(img).unsqueeze(0).float()
        x /= 255

        return x, y, file_name


class CombinedSegGrazPedWriDataset(Dataset):
    """
    Dataset that combines dataset with ground truth and dataset with pseudo label. Retuns all files from dataset with ground truth,
    but for each file from dataset with ground truth, a random file from dataset with pseudo label is returned.
    """
    # calculated over training split
    IMG_MEAN = LightSegGrazPedWriDataset.IMG_MEAN
    IMG_STD = LightSegGrazPedWriDataset.IMG_STD
    # copy all attributes from LightSegGrazPedWriDataset
    BONE_LABEL = LightSegGrazPedWriDataset.BONE_LABEL
    BONE_LABEL_MAPPING = {k: v for k, v in zip(BONE_LABEL, range(len(BONE_LABEL)))}
    N_CLASSES = len(BONE_LABEL)

    def __init__(self, ds_with_gt: LightSegGrazPedWriDataset, ds_with_pseudo_lbl: SavedSegGrazPedWriDataset):
        super().__init__()
        self.ds_with_gt = ds_with_gt
        self.ds_with_pseudo_lbl = ds_with_pseudo_lbl

    def __len__(self):
        return len(self.ds_with_gt)

    def __getitem__(self, index):
        x1, y1, file_name1 = self.ds_with_gt[index]
        # get random index from dataset with pseudo label
        rnd_idx = randint(0, len(self.ds_with_pseudo_lbl) - 1)
        x2, y2, file_name2 = self.ds_with_pseudo_lbl[rnd_idx]

        return {'gt': (x1, y1, file_name1), 'pseudo_lbl': (x2, y2, file_name2)}


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from numpy import ma

    ds = SavedSegGrazPedWriDataset('data/seg_masks/29f483c5ab6d4f2991f96958d6c68b1a/sam_box_refine_pos_points_neg_points_500.h5', True)
    #ds = LightSegGrazPedWriDataset('test', number_training_samples=50)
    print(f'Number of classes: {ds.N_CLASSES}')
    idx = randint(0, len(ds) - 1)
    x, y, filename = ds[0]
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(x.squeeze(0), cmap='gray')
    ax[1].imshow(x.squeeze(0), cmap='gray')
    ax[1].imshow(y.argmax(0), alpha=y.any(0).float() * 0.5)
    for lbl, mask in zip(ds.BONE_LABEL, y):
        plt.figure(filename)
        plt.imshow(x.squeeze(0), cmap='gray')
        plt.imshow(ma.masked_where(mask == 0, mask), alpha=0.5)
        plt.title(lbl)
    plt.show()
