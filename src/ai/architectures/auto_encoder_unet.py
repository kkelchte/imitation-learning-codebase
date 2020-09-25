#!/bin/python3.8
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ai.base_net import ArchitectureConfig, BaseNet
from src.ai.architectures.auto_encoder_conv1 import Net as BaseAENet
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension

"""
Four encoding and four decoding layers with dropout.
Expects 1x64x64 inputs and outputs 64x64
"""


class Net(BaseAENet):

    def __init__(self, config: ArchitectureConfig, quiet: bool = False):
        super().__init__(config=config, quiet=True)
        self._logger = get_logger(name=get_filename_without_extension(__file__),
                                  output_path=config.output_path,
                                  quiet=False)
        if not quiet:
            self.h = 64
            self.in_conv = DoubleConv(1, self.h)
            self.down1 = Down(self.h, 2 * self.h)
            self.down2 = Down(2 * self.h, 4 * self.h)
            self.down3 = Down(4 * self.h, 8 * self.h)
            self.down4 = Down(8 * self.h, (8 if not self.vae else 16) * self.h)

            self.out_conv = OutConv(self.h, 1)
            self.up1 = Up(2 * self.h, self.h, bilinear=True)
            self.up2 = Up(4 * self.h, self.h, bilinear=True)
            self.up3 = Up(8 * self.h, 2 * self.h, bilinear=True)
            self.up4 = Up(16 * self.h, 4 * self.h, bilinear=True)

            self.initialize_architecture()
            cprint(f'Started.', self._logger)

    def forward_with_distribution(self, inputs, train: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        return network outputs and latent distribution
        """
        inputs = self.process_inputs(inputs=inputs, train=train)
        if self._config.finetune:
            with torch.no_grad():
                x1 = self.in_conv(inputs)
                x2 = self.down1(x1)
                x3 = self.down2(x2)
                x4 = self.down3(x3)
                x5 = self.down4(x4)
        else:
            x1 = self.in_conv(inputs)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)

        if self.dropout is not None and train:
            x5 = self.dropout(x5)

        if self.vae:
            mean = x5[:, :8 * self.h]
            std = torch.exp(x5[:, 8 * self.h:]/2)
            x5 = mean + std * torch.rand_like(std)
        else:
            mean = None
            std = None

        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.out_conv(x).squeeze(dim=1)

        return x, mean, std

    def forward(self, inputs, train: bool = False) -> torch.Tensor:
        x, _, _ = self.forward_with_distribution(inputs, train)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
