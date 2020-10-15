from typing import Tuple, Type

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(torch.nn.Module):

    def __init__(self, input_channels,
                 output_channels,
                 batch_norm: bool = True,
                 activation: torch.nn.ReLU = torch.nn.ReLU(),
                 pool: torch.nn.MaxPool2d = None,
                 strides: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (0, 0),
                 kernel_sizes: Tuple[int, int] = (3, 3)):
        """
        Create a residual block
        :param input_channels: number of input channels at input
        :param output_channels: number of input channels at input
        :param batch_norm: bool specifying to use batch norm 2d (True)
        :param activation: specify torch nn module activation (ReLU)
        :param pool: specify pooling layer applied as first layer
        :param strides: tuple specifying the stride and so the down sampling
        """
        super().__init__()
        self._down_sample = torch.nn.Conv2d(input_channels, output_channels, kernel_size=1,
                                            stride=sum([s - 1 for s in strides]) + (1 if pool is None else 2)) \
            if sum([s - 1 for s in strides]) > 0 or pool is not None else torch.nn.Identity()
        self._final_activation = activation
        elements = []
        if pool is not None:
            elements.append(pool)
        elements.append(torch.nn.Conv2d(in_channels=input_channels,
                                        out_channels=output_channels,
                                        kernel_size=kernel_sizes[0],
                                        padding=padding[0],
                                        stride=strides[0]))
        if batch_norm:
            elements.append(torch.nn.BatchNorm2d(output_channels))
        elements.append(activation)
        elements.append(torch.nn.Conv2d(in_channels=output_channels,
                                        out_channels=output_channels,
                                        kernel_size=kernel_sizes[1],
                                        padding=padding[1],
                                        stride=strides[1]))
        if batch_norm:
            elements.append(torch.nn.BatchNorm2d(output_channels))
        elements.append(activation)
        self.residual_net = torch.nn.Sequential(*elements)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.residual_net(inputs)
        x += self._down_sample(inputs)
        return self._final_activation(x)


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
