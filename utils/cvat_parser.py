from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage.draw import polygon2mask
from xmltodict import parse


@dataclass
class Mask:
    """Mask for a single anatomy"""
    mask: np.ndarray
    anatomy: str


class CVATParser:
    def __init__(self, path2xml, merge_same_anatomy: bool, create_bone_mask: bool, drop_sonstiges: bool, mask_dtype=np.uint8):
        """
        Parse Mask from CVAT xml file. Currently only supports mask format of type 'polygon' and 'mask'
        :param path2xml: CVAT xml file (Version Image 1.1) or list of filespk
        :param merge_same_anatomy: if True, masks with same anatomy are merged, set False if instance segmentation is needed
        :param create_bone_mask: if True, create mask for bone (Ossis) merging all other masks
        :param drop_sonstiges: if True, drop all masks with anatomy 'Sonstiges'
        :param mask_dtype: dtype of mask
        """
        super().__init__()
        self.merge_same_anatomy = merge_same_anatomy
        self.create_bone_mask = create_bone_mask
        self.drop_sonstiges = drop_sonstiges
        self.mask_dtype = mask_dtype
        self.img_dicts = list()

        if not isinstance(path2xml, list):
            path2xml = [path2xml]
        for xml in path2xml:
            with open(xml) as fd:
                self.img_dicts.extend(parse(fd.read())['annotations']['image'])
        self.name2id_map = {img['@name'].split('.')[0]: i for i, img in enumerate(self.img_dicts)}
        # handle cases if only one mask or polygon is present
        for img_dict in self.img_dicts:
            if 'mask' in img_dict:
                if isinstance(img_dict['mask'], dict):
                    img_dict['mask'] = [img_dict['mask']]
            if 'polygon' in img_dict:
                if isinstance(img_dict['polygon'], dict):
                    img_dict['polygon'] = [img_dict['polygon']]

    @classmethod
    def cvt_mask_list_2_dict(cls, mask_list: list[Mask]) -> dict:
        """
        Convert list of masks to dictionary with anatomy as key and mask ndarray as value
        :param mask_list: list of masks
        :return:
        """
        return {mask.anatomy: mask.mask for mask in mask_list}

    def extract_masks(self, img_name: str) -> list[Mask]:
        """
        Extract masks for a single image
        :param img_name: filename of image
        :return: list of masks
        """
        try:
            img_dict = self.img_dicts[self.name2id_map[img_name]]
        except KeyError:
            raise KeyError(f'Image {img_name} not found in annotations')

        img_h = int(img_dict['@height'])
        img_w = int(img_dict['@width'])

        masks = []
        if 'mask' in img_dict:
            for mask_dict in img_dict['mask']:
                masks.append(self.get_mask_from_mask(mask_dict, img_h, img_w))

        if 'polygon' in img_dict:
            for polygon_dict in img_dict['polygon']:
                masks.append(self.get_mask_from_polygone(polygon_dict, img_h, img_w))

        assert len(masks) > 0, f'No masks found for image {img_name}'
        assert all([mask.mask.shape == (img_h, img_w) for mask in masks]), 'Mask and image shape do not match'
        assert all([mask.mask.dtype == self.mask_dtype for mask in masks]), 'Mask dtype does not match'

        # merge masks with same anatomy, else return list of instance masks
        if self.merge_same_anatomy:
            anatomy_dict = {}
            for mask in masks:
                if mask.anatomy in anatomy_dict:
                    anatomy_dict[mask.anatomy] |= mask.mask
                else:
                    anatomy_dict[mask.anatomy] = mask.mask
            masks = self._dict2mask_list(anatomy_dict)

        if self.create_bone_mask:
            anatomy_dict = {'Ossis': np.zeros((img_h, img_w), dtype=self.mask_dtype)}
            for mask in masks:
                anatomy_dict[mask.anatomy] = mask.mask
                anatomy_dict['Ossis'] |= mask.mask
            masks = self._dict2mask_list(anatomy_dict)

        if self.drop_sonstiges:
            masks = list(filter(lambda x: x.anatomy != 'Sonstiges', masks))

        return masks

    def _dict2mask_list(self, d: dict) -> list[Mask]:
        """
        Convert dictionary to list of masks
        :param d: dictionary with anatomy as key and mask ndarray as value
        :return:
        """
        return [Mask(mask, anatomy) for anatomy, mask in d.items()]

    def _rle2mask(self, rle: list[int], width: int, height: int) -> np.ndarray:
        # https://github.com/opencv/cvat/issues/5828
        decoded = [0] * (width * height)  # create bitmap container
        decoded_idx = 0
        value = False

        for v in rle:
            decoded[decoded_idx:decoded_idx + v] = [value] * v
            decoded_idx += v
            value = not value  # invert value for next run

        decoded = np.array(decoded, dtype=self.mask_dtype)
        decoded = decoded.reshape((height, width))  # reshape to mask size

        return decoded

    def get_mask_from_mask(self, mask_dict: dict, img_h: int, img_w: int) -> Mask:
        # get attributes from mask_dict
        mask_pos_top = int(mask_dict['@top'])
        mask_pos_left = int(mask_dict['@left'])
        mask_w = int(mask_dict['@width'])
        mask_h = int(mask_dict['@height'])
        rle = list(map(int, mask_dict['@rle'].split(',')))

        # decode rle encoding to mask
        mask = self._rle2mask(rle, int(mask_dict['@width']), int(mask_dict['@height']))
        mask_img = np.zeros((img_h, img_w), dtype=self.mask_dtype)
        mask_img[mask_pos_top:mask_pos_top + mask_h, mask_pos_left:mask_pos_left + mask_w] = mask

        return Mask(mask_img, mask_dict['attribute']['#text'])

    def get_mask_from_polygone(self, polygone_dict: dict, img_h: int, img_w: int) -> Mask:
        # get attributes from polygone_dict
        pnts = polygone_dict['@points']

        # convert points to numpy array
        pnts = pnts.split(';')
        pnts = list(map(lambda x: x.split(','), pnts))
        pnts = list(map(lambda x: list(map(float, x)), pnts))
        pnts_array = np.array(pnts)
        pnts_array = np.flip(pnts_array, axis=1)

        # convert points to mask
        mask = polygon2mask((img_h, img_w), pnts_array)
        mask = mask.astype(self.mask_dtype)

        return Mask(mask, polygone_dict['attribute']['#text'])

    @property
    def available_file_names(self) -> list[str]:
        return list(self.name2id_map.keys())


if __name__ == '__main__':
    # test
    import matplotlib.pyplot as plt
    from numpy import ma

    img_path = Path('datasets/data/GRAZPEDWRI-DX/img8bit')
    img_file = '0208_1044966620_01_WRI-L1_F003'

    parser = CVATParser(['datasets/data/GRAZPEDWRI-DX/annotation_cvat/xml_files/annotations_train1.xml'],
                        True,
                        False,
                        False)
    masks = parser.extract_masks(img_file)
    img = plt.imread(img_path.joinpath(img_file).with_suffix('.png'))
    for mask in masks:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img)
        ax[1].imshow(img, 'gray')
        ax[1].imshow(ma.masked_where(mask.mask == 0, mask.mask), alpha=0.5)
        ax[0].set_title('Image')
        ax[1].set_title(mask.anatomy)
    plt.show()
