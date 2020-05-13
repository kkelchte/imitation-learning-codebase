#!/bin/python3.7

import torch
import torch.nn as nn

from src.ai.base_net import BaseNet, ArchitectureConfig
from src.ai.utils import mlp_creator
from src.core.data_types import Action
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension

"""
Tiny four encoding and three decoding layers with dropout.
Expects 3x128x128 inputs and outputs 1c 
"""


class Net(BaseNet):

    def __init__(self, config: ArchitectureConfig, quiet: bool = False):
        super().__init__(config=config, quiet=True)
        self._logger = get_logger(name=get_filename_without_extension(__file__),
                                  output_path=config.output_path,
                                  quiet=False)
        if not quiet:
            cprint(f'Started.', self._logger)
        self.input_size = (3, 128, 128)
        self.output_size = (6,)
        self.discrete = False
        self.dropout = nn.Dropout(p=config.dropout) if config.dropout != 'default' else None
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.Conv2d(128, 256, 4, stride=2),
        )
        self.decoder = mlp_creator(sizes=[256 * 6 * 6, 128, 128, self.output_size[0]],
                                   activation=nn.ReLU,
                                   output_activation=nn.Tanh)
        self.load_network_weights()

    def forward(self, inputs, train: bool = False) -> torch.Tensor:
        """
        Outputs steering action only
        """
        inputs = super().forward(inputs=inputs, train=train)
        x = self.encoder(inputs)
        x = x.flatten(start_dim=1)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.decoder(x)
        return x

    def get_action(self, inputs, train: bool = False) -> Action:
        output = self.forward(inputs, train=train)
        return Action(actor_name=get_filename_without_extension(__file__),
                      value=output.data)

