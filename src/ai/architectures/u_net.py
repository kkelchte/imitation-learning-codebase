#!/bin/python3.8
"""
U-net implementation
Source: https://github.com/milesial/Pytorch-UNet
"""
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
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear=True)
        self.up2 = Up(512, 256 // factor, bilinear=True)
        self.up3 = Up(256, 128 // factor, bilinear=True)
        self.up4 = Up(128, 64, bilinear=True)
        self.outc = OutConv(64, 1)
        self.initialize_architect()

    def forward(self, inputs, train: bool = False) -> torch.Tensor:
        inputs = self.process_inputs(inputs=inputs, train=train)

        x1 = self.inc(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.dropout is not None and train:
            x5 = self.dropout(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x.squeeze(dim=1)

    def get_action(self, inputs, train: bool = False) -> Action:
        raise NotImplementedError
