from pathlib import Path

import albumentations as A
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from scripts.cvat_parser import CVATParser


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

    def __init__(self, mode: str, rescale_HW: tuple = (384, 224)):
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

    ds = LightSegGrazPedWriDataset('train')
    idx = 1  # randint(0, len(ds) - 1)
    x, y, filename = ds[idx]
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(x.squeeze(0), cmap='gray')
    ax[1].imshow(y.argmax(0))
    for lbl, mask in zip(ds.BONE_LABEL, y):
        plt.figure()
        plt.imshow(x.squeeze(0), cmap='gray')
        plt.imshow(ma.masked_where(mask == 0, mask), alpha=0.5)
        plt.title(lbl)
    plt.show()
