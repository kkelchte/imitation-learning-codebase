#!/bin/python3.8
"""
U-net implementation
Source: https://github.com/milesial/Pytorch-UNet
"""
from typing import Tuple

import torch
import torch.nn as nn

from src.ai.architectures.bc_variational_auto_encoder import Net as BaseNet
from src.ai.base_net import ArchitectureConfig
from src.ai.architectural_components import Down, DoubleConv, Up, OutConv
from src.core.data_types import Action
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension


""" Full assembly of the parts to form the complete network """


class Net(BaseNet):

    def __init__(self, config: ArchitectureConfig, quiet: bool = False):
        super().__init__(config=config, quiet=True)
        self.input_size = (1, 64, 64)
        self.output_size = (64, 64)

        if not quiet:
            self.h = 512
            self._logger = get_logger(name=get_filename_without_extension(__file__),
                                      output_path=config.output_path,
                                      quiet=False)
            self.inc = DoubleConv(1, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            self.down4 = Down(512, self.h * (2 if self.vae else 1))
            self.up1 = Up(self.h * 2, 256, bilinear=True)
            self.up2 = Up(512, 128, bilinear=True)
            self.up3 = Up(256, 64, bilinear=True)
            self.up4 = Up(128, 64, bilinear=True)
            self.outc = OutConv(64, 1)
            self.initialize_architecture()
            cprint(f'Started.', self._logger)

    def _encode(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                     torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self.process_inputs(inputs=inputs)
        if self._config.finetune:
            with torch.no_grad():
                x1 = self.inc(inputs)
                x2 = self.down1(x1)
                x3 = self.down2(x2)
                x4 = self.down3(x3)
                x5 = self.down4(x4)
        else:
            x1 = self.inc(inputs)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
        if self.dropout is not None:
            x5 = self.dropout(x5)
        return x1, x2, x3, x4, x5

    def _decode(self, inputs: Tuple[torch.Tensor, torch.Tensor,
                                    torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x1, x2, x3, x4, x5 = inputs
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x.squeeze(dim=1)

    def _forward_with_distribution(self, inputs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        return network outputs and latent distribution
        """
        x1, x2, x3, x4, x5 = self._encode(inputs)
        mean = x5[:, :self.h]
        std = torch.exp(x5[:, self.h:]/2)
        x5 = mean + std * torch.rand_like(std)
        x = self._decode((x1, x2, x3, x4, x5))
        return x, mean, std

    def get_action(self, inputs, train: bool = False) -> Action:
        raise NotImplementedError
