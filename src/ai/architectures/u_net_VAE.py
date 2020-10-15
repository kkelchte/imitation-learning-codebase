#!/bin/python3.8
"""
U-net implementation
Source: https://github.com/milesial/Pytorch-UNet
"""
from typing import Tuple

import torch
import torch.nn as nn

from src.ai.base_net import BaseNet, ArchitectureConfig
from src.ai.architectural_components import Down, DoubleConv, Up
from src.core.data_types import Action
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension


""" Full assembly of the parts to form the complete network """


class Net(BaseNet):

    def __init__(self, config: ArchitectureConfig, quiet: bool = False):
        super().__init__(config=config, quiet=True)
        self._logger = get_logger(name=get_filename_without_extension(__file__),
                                  output_path=config.output_path,
                                  quiet=False)
        if not quiet:
            cprint(f'Started.', self._logger)
        self.input_size = (1, 64, 64)
        self.input_scope = 'default'
        self.output_size = (64, 64)
        self.discrete = False
        self.dropout = nn.Dropout(p=config.dropout) if config.dropout != 'default' else None

        self.inc = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        factor = 2
        self.down4 = Down(512, 2*1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear=True)
        self.up2 = Up(512, 256 // factor, bilinear=True)
        self.up3 = Up(256, 128 // factor, bilinear=True)
        self.up4 = Up(128, 64, bilinear=True)
        self.outc = OutConv(64, 1)
        self.initialize_architecture()

    def forward_with_distribution(self, inputs, train: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Outputs steering action only
        """
        inputs = super().forward(inputs=inputs, train=train)

        x1 = self.inc(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        mean = x5[:, :512]
        std = torch.exp(x5[:, 512:]/2)
        x5 = mean + std * torch.rand_like(std)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x).squeeze(dim=1)
        return x, mean, std

    def forward(self, inputs, train: bool = False) -> torch.Tensor:
        x, mean, std = self.forward_with_distribution(inputs, train)
        return x

    def get_action(self, inputs, train: bool = False) -> Action:
        raise NotImplementedError
