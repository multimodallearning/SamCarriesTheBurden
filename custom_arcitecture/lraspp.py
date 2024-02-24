from torchvision.models.segmentation import LRASPPHead
from torch import nn
from custom_arcitecture.modelio import LoadableModel

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

