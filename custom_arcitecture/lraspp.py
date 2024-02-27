from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from custom_arcitecture.modelio import LoadableModel, store_config_args
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


class BasicBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, downsample: bool = False):
        super(BasicBlock, self).__init__()
        self.relu = nn.LeakyReLU(inplace=True)
        self.features = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False, stride=2 if downsample else 1),
            nn.InstanceNorm2d(out_channel, affine=True),
            self.relu,

            nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channel, affine=True),
        )
        if in_channel == out_channel and not downsample:
            self.identity_projection = nn.Identity()
        else:
            self.identity_projection = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, bias=False, stride=2 if downsample else 1),
                nn.InstanceNorm2d(out_channel, affine=True)
            )

    def forward(self, x):
        identity = self.identity_projection(x)
        features = self.features(x)
        activation = self.relu(identity + features)
        return activation


# adapted from torchvision/models/segmentation/lraspp.py
class LRASPPHead(nn.Module):

    def __init__(self, low_channels: int, high_channels: int, num_classes: int, inter_channels: int) -> None:
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.InstanceNorm2d(inter_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(low_channels, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)

    def forward(self, input: Dict[str, torch.Tensor]) -> torch.Tensor:
        low = input["low"]
        high = input["high"]

        x = self.cbr(high)
        s = self.scale(high)
        x = x * s
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)

        return self.low_classifier(low) + self.high_classifier(x)


class LRASPPOnSAM(LoadableModel):
    @store_config_args
    def __init__(self, n_classes: int, n_last_channel: int = 64, model_type: str = "vit_h",
                 sam_checkpoint: str = "data/sam_vit_h_4b8939.pth"):
        super().__init__()
        self.sam_img_encoder = sam_model_registry[model_type](checkpoint=sam_checkpoint).image_encoder
        self.sam_img_encoder.pos_embed = None
        self.transform_for_sam = ResizeLongestSide(self.sam_img_encoder.img_size).apply_image_torch
        self.res_encoder = nn.Sequential(
            BasicBlock(256, 512, downsample=True),
            BasicBlock(512, 512, downsample=False)
        )
        self.head = LRASPPHead(256, 512, n_classes, n_last_channel)

        # img statistics
        self.IMG_MEAN = 0.3505533917353781
        self.IMG_STD = 0.22763733675869177

    def forward(self, x) -> torch.Tensor:
        # reverse z-normalization, because the sam_img_encoder expects the input to be in the range [0, 1]
        x = x * self.IMG_STD + self.IMG_MEAN

        # grayscale to RGB
        x = x.expand(-1, 3, -1, -1)
        original_size = x.shape[-2:]

        with torch.no_grad():
            x = self.transform_for_sam(x)
            x_low = self.sam_img_encoder(x)
        x_high = self.res_encoder(x_low)
        x = self.head({"low": x_low, "high": x_high})

        # scale up to the original size
        x = F.interpolate(x, size=original_size, mode="bilinear", align_corners=False)
        return x

    def parameters(self, recurse: bool = True):
        # only train the res_encoder and head
        params = list(self.res_encoder.parameters(recurse)) + list(self.head.parameters(recurse))
        iterator = iter(params)
        return iterator

    def train(self, mode: bool = True):
        super().train(mode)
        # freeze the sam_img_encoder
        self.sam_img_encoder.eval()


if __name__ == '__main__':
    from scripts.seg_grazpedwri_dataset import LightSegGrazPedWriDataset
    from torchinfo import summary

    model = LRASPPOnSAM(n_classes=17)
    ds = LightSegGrazPedWriDataset('train')
    x, y, _ = ds[0]
    p = model.parameters()
    y_hat = model(x.unsqueeze(0))
    summary(model, input_size=(1, 1, 384, 224))
