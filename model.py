import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


def init_weights(module):
    if isinstance(module, nn.Conv3d):
        nn.init.xavier_uniform_(module.weight)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=1, features=None):
        super(UNet, self).__init__()
        if features is None:
            features = [16, 32, 64, 128]
        self.down_samples = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        """Initialise layers for down sampling and up sampling"""
        ###  Down Sampling  ##
        for feature in features:
            self.down_samples.append(
                DoubleConv(in_channels, feature)
            )
            in_channels = feature

        # initialise weights
        self.down_samples.apply(init_weights)

        ###  Up Sampling  ###
        for feature in reversed(features):
            self.up_samples.append(
                nn.ConvTranspose3d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.up_samples.append(DoubleConv(feature*2, feature))

        # initialise weights
        self.up_samples.apply(init_weights)

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.down_samples:
            x = down(x)
            skip_connections.append(x)
            self.pool(x)

        x = self.bottleneck(x)
        # reverse skip connections
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.up_samples), 2):
            print(f"x (before): {x.shape}")
            x = self.up_samples[idx](x)
            print(f"x (upsampled): {x.shape}")
            skip_connection = skip_connections[idx//2]

            # make sure sizes match
            if x.shape != skip_connection.shape:
                print(f"index: {idx}")
                print(f"up sample: {len(self.up_samples)}")
                print(f"skip connections: {skip_connection.shape}")
                x = TF.resize(x, size=[155, 64, 64], antialias=True)
                print(f"x (resized): {x.shape}")

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.up_samples[idx+1](concat_skip)

        return self.final_conv(x)


def test():
    x = torch.randn((8, 4, 155, 64, 64))
    model = UNet(in_channels=4)
    prediction = model(x)
    assert prediction.shape == x.shape

