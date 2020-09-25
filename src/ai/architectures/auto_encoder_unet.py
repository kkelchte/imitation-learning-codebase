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
            self.up4 = Up(2 * self.h, self.h, bilinear=True)
            self.up3 = Up(4 * self.h, self.h, bilinear=True)
            self.up2 = Up(8 * self.h, 2 * self.h, bilinear=True)
            self.up1 = Up(16 * self.h, 4 * self.h, bilinear=True)

            self.initialize_architecture()
            cprint(f'Started.', self._logger)

    def forward_with_distribution(self, inputs, train: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        return network outputs and latent distribution
        """
        outputs = self._encode(inputs, train)
        last_key = list(outputs.keys())[-1]
        mean = outputs[last_key][:, :8 * self.h]
        std = torch.exp(outputs[last_key][:, 8 * self.h:]/2)
        outputs[last_key] = mean + std * torch.rand_like(std)
        x = self._decode(outputs)
        return x, mean, std

    def _decode(self, inputs: dict) -> torch.Tensor:
        x = self.up1(inputs['down4'], inputs['down3'])
        x = self.up2(x, inputs['down2'])
        x = self.up3(x, inputs['down1'])
        x = self.up4(x, inputs['in_conv'])
        return self.out_conv(x).squeeze(dim=1)

    def _encode(self, inputs, train: bool = False) -> dict:
        inputs = self.process_inputs(inputs=inputs, train=train)

        def _extract(x: torch.Tensor) -> dict:
            outputs = {'in_conv': self.in_conv(x)}
            outputs['down1'] = self.down1(outputs['in_conv'])
            outputs['down2'] = self.down2(outputs['down1'])
            outputs['down3'] = self.down3(outputs['down2'])
            outputs['down4'] = self.down4(outputs['down3'])
            return outputs

        if self._config.finetune:
            with torch.no_grad():
                outputs = _extract(inputs)
        else:
            outputs = _extract(inputs)

        if self.dropout is not None and train:
            last_key = list(outputs.keys())[-1]
            outputs[last_key] = self.dropout(outputs[last_key])
        return outputs


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