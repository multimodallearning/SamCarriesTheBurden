import json
import logging

import h5py
import torch
from torch.utils.data import Dataset
import cv2
from pathlib import Path
from tqdm import tqdm
from torch.nn import functional as F


class DentalDataset(Dataset):
    SPLIT_IDX = 450
    CLASS_LABEL = ['13', '14', '15', '11', '12', '19', '20', '21', '22', '23', '24', '25', '27', '32', '16', '26', '17',
                   '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '18', '28', '29', '30', '31']
    N_CLASSES = len(CLASS_LABEL)
    BCE_POS_WEIGHTS = torch.tensor([270.1158, 192.5302, 188.6575, 189.5051, 244.8739, 214.9675, 222.4272,
                                    227.7497, 185.8342, 259.1098, 308.6378, 301.8164, 179.1835, 265.5014,
                                    344.5692, 255.3629, 284.0897, 314.7787, 177.8944, 176.8611, 257.4408,
                                    246.6468, 183.3314, 222.9948, 170.8807, 168.2754, 226.9026, 188.9870,
                                    212.7988, 211.4843, 192.7491, 173.3434])
    BCE_POS_WEIGHTS = BCE_POS_WEIGHTS.view(N_CLASSES, 1, 1).expand(N_CLASSES, 224, 384)
    IMG_MEAN = 0.3432
    IMG_STD = 0.2211

    def __init__(self, mode: str, number_training_samples: int | str = 'all'):
        super().__init__()

        # load data
        img_dir = Path('data/DentalSeg/img')
        mask_dir = Path('data/DentalSeg/masks_machine')
        self.available_files = sorted([f.stem for f in img_dir.glob('*.jpg')])

        # select images for training or testing
        if mode == 'train':

            if number_training_samples == 'all':
                number_training_samples = self.SPLIT_IDX
            elif number_training_samples > self.SPLIT_IDX:
                raise ValueError(
                    f'number_training_samples {number_training_samples} is greater than the available samples '
                    f'for training {self.SPLIT_IDX}')

            self.available_files = self.available_files[:number_training_samples]
        elif mode == 'val':
            self.available_files = self.available_files[self.SPLIT_IDX:][:75]
        elif mode == 'test':
            self.available_files = self.available_files[self.SPLIT_IDX:][75:]
        elif mode == 'all':
            logging.warning('Using all data. Please do not use this mode in context of training.')
        else:
            raise ValueError(f'Unknown mode {mode}')

        # load images and masks
        self.data = {}
        for file_name in tqdm(self.available_files, desc='Loading data', unit='file'):
            img = cv2.imread(str(img_dir / (file_name + '.jpg')), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(str(mask_dir / (file_name + '.png')), cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                raise FileNotFoundError(f'Image or mask not found for file {file_name}')
            if img.shape != mask.shape:
                raise ValueError(f'Image and mask shape mismatch for file {file_name}')

            # resize
            img = cv2.resize(img, (384, 224))
            mask = cv2.resize(mask, (384, 224), interpolation=cv2.INTER_NEAREST)

            img = torch.from_numpy(img).float() / 255
            img = img.unsqueeze(0)  # add channel dimension
            # one hot encode mask and remove background class
            mask = F.one_hot(torch.from_numpy(mask).long(), num_classes=self.N_CLASSES + 1).permute(2, 0, 1)[1:].bool()

            self.data[file_name] = (img, mask)

    def __len__(self):
        return len(self.available_files)

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        filestem = self.available_files[idx]
        img, mask = self.data[filestem]

        return img, mask.float(), filestem


class SavedDentalDataset(Dataset):
    CLASS_LABEL = DentalDataset.CLASS_LABEL
    N_CLASSES = len(CLASS_LABEL)
    BCE_POS_WEIGHTS = DentalDataset.BCE_POS_WEIGHTS
    IMG_MEAN = DentalDataset.IMG_MEAN
    IMG_STD = DentalDataset.IMG_STD

    def __init__(self, h5_file: Path):
        super().__init__()
        h5_file = h5py.File(h5_file, 'r')
        lbl_loaded = json.loads(h5_file.attrs['labels'])
        assert list(lbl_loaded.keys()) == self.CLASS_LABEL, 'Loaded labels do not match'

        # load data meta and other information
        self.img_path = Path('data/DentalSeg/img')
        self.ds_saved_seg = h5_file['segmentation_mask']

        # get file names
        self.available_file_names = list(self.ds_saved_seg.keys())

        # init transformation
        self.resize_lbl = lambda x: F.interpolate(x.float().unsqueeze(0), size=(224, 384), mode='nearest').squeeze(0)

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
        img = cv2.imread(str(self.img_path.joinpath(file_name).with_suffix('.jpg')), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, y.shape[-2:][::-1], interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(img).unsqueeze(0).float()
        x /= 255

        return x, y, file_name


class MeanTeacherDentalDataset(Dataset):
    CLASS_LABEL = DentalDataset.CLASS_LABEL
    N_CLASSES = len(CLASS_LABEL)
    BCE_POS_WEIGHTS = DentalDataset.BCE_POS_WEIGHTS
    IMG_MEAN = DentalDataset.IMG_MEAN
    IMG_STD = DentalDataset.IMG_STD

    def __init__(self, number_training_samples: int | str = 'all', model_id_pseudo_label: str = None,
                 dsc_agreement_threshold: float = None):
        """
        Dataset that combines dataset with and dataset without ground truth.
        """
        super().__init__()
        self.img_path = Path('data/DentalSeg/img')

        self.ds_with_gt = DentalDataset('train', number_training_samples)
        self.unlabeled_files_names = self.img_path.rglob('*.jpg')
        self.unlabeled_files_names = [f.stem for f in self.unlabeled_files_names]
        self.unlabeled_files_names = list(set(self.unlabeled_files_names) - set(self.ds_with_gt.available_files))
        assert len(set(self.unlabeled_files_names) & set(self.ds_with_gt.available_files)) == 0, 'Files are duplicated'

        self.available_file_names = self.ds_with_gt.available_files + self.unlabeled_files_names

        # add reliability pseudo label
        self.use_pseudo_label = False
        if model_id_pseudo_label is not None and dsc_agreement_threshold is not None:
            self.use_pseudo_label = True
            pseudo_label_path = Path('data/seg_masks').joinpath(model_id_pseudo_label).joinpath(
                f'selected_pseudo_labels_450_dsc_{str(dsc_agreement_threshold).replace(".", "")}.h5')
            assert pseudo_label_path.exists(), f'Pseudo label file does not exist. Please check the path: {pseudo_label_path}'
            self.ds_with_pseudo_lbl = SavedDentalDataset(pseudo_label_path)
            assert all([f in self.available_file_names for f in self.ds_with_pseudo_lbl.available_file_names]), \
                'Pseudo label files are not in available files'

        # load images without ground truth
        self.unlabeled_imgs = {}
        for file_name in tqdm(self.unlabeled_files_names, desc='Loading unlabeled data', unit='file'):
            img = cv2.imread(str(self.img_path.joinpath(file_name).with_suffix('.jpg')), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (384, 224), interpolation=cv2.INTER_LINEAR)
            img = torch.from_numpy(img).unsqueeze(0).float()
            img /= 255
            self.unlabeled_imgs[file_name] = img

    def __len__(self):
        return len(self.available_file_names)

    def __getitem__(self, index):
        file_name = self.available_file_names[index]

        if file_name in self.ds_with_gt.available_files:  # gt available
            return self.ds_with_gt[self.ds_with_gt.available_files.index(file_name)]
        else:  # no gt available
            return self.unlabeled_imgs[file_name], None, file_name


if __name__ == '__main__':
    ds = MeanTeacherDentalDataset(45)
