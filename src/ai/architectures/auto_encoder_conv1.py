#!/bin/python3.8
from typing import Tuple

import torch
import torch.nn as nn

from src.ai.base_net import BaseNet, ArchitectureConfig
from src.ai.utils import mlp_creator
from src.core.data_types import Action
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension

"""
Four encoding and four decoding layers with dropout.
Expects 1x64x64 inputs and outputs 64x64
"""


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
        self._config.batch_normalisation = False if config.batch_normalisation == 'default' \
            else config.batch_normalisation
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, 1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 1, 1, stride=1),
        )

        self.initialize_architecture()

    def forward(self, inputs, train: bool = False) -> torch.Tensor:
        """
        Outputs steering action only
        """
        inputs = super().forward(inputs=inputs, train=train)
        return self.network(inputs).squeeze(dim=1)

    def get_action(self, inputs, train: bool = False) -> Action:
        output, _, _ = self.forward(inputs, train=train)
        return Action(actor_name=get_filename_without_extension(__file__),
                      value=output.data)

