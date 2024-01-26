import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.utils.prune as prune


def init_weights(module):
    if isinstance(module, nn.Conv3d):
        nn.init.xavier_uniform_(module.weight)


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=1, features=None):
        super(UNet, self).__init__()
        if features is None:
            features = [64, 128, 256]

        # Down sampling
        self.down1 = double_conv(in_channels, features[0])
        self.down2 = double_conv(features[0], features[1])
        self.down3 = double_conv(features[1], features[2])
        self.maxpool = nn.MaxPool3d(2)

        # Up sampling
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.up1 = double_conv(features[1]+features[2], features[1])
        self.up2 = double_conv(features[0]+features[1], features[0])

        self.final_conv = nn.Conv3d(features[0], out_channels, 1)

    def forward(self, x):
        # down sampling
        conv1 = self.down1(x)
        x = self.maxpool(conv1)

        conv2 = self.down2(x)
        x = self.maxpool(conv2)

        x = self.down3(x)

        # up sampling
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.up1(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.up2(x)

        return self.final_conv(x)

    def prune_network(self):
        prune.random_structured(self.down1, name="weight", amount=0.3, dim=0)
        prune.random_structured(self.down2, name="weight", amount=0.3, dim=0)
        prune.random_structured(self.down3, name="weight", amount=0.3, dim=0)
        prune.random_structured(self.upsample, name="weight", amount=0.3, dim=0)
        prune.random_structured(self.up1, name="weight", amount=0.3, dim=0)
        prune.random_structured(self.up2, name="weight", amount=0.3, dim=0)
        prune.random_structured(self.final_conv, name="weight", amount=0.3, dim=0)


def test():
    x = torch.randn((8, 4, 155, 64, 64))
    model = UNet(in_channels=4)
    prediction = model(x)
    assert prediction.shape == x.shape

