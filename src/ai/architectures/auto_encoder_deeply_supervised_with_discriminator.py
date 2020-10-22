#!/bin/python3.8
from typing import Tuple

import torch
import torch.nn as nn

from src.ai.architectural_components import ResidualBlock
from src.ai.architectures.auto_encoder_deeply_supervised import Net as BaseNet
from src.ai.base_net import ArchitectureConfig
from src.ai.utils import mlp_creator
from src.core.data_types import Action
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension

"""
Four encoding and four decoding layers with dropout.
Expects 3x200x200 inputs and outputs 200x200
"""


class Net(BaseNet):

    def __init__(self, config: ArchitectureConfig, quiet: bool = False):
        super().__init__(config=config, quiet=True)

        if not quiet:
            self._logger = get_logger(name=get_filename_without_extension(__file__),
                                      output_path=config.output_path,
                                      quiet=False)
            self._discriminator = mlp_creator(sizes=[self.output_size, 64, 64, 1],
                                              activation=nn.ReLU(),
                                              output_activation=nn.ReLU())
            self.initialize_architecture()
            cprint(f'Started.', self._logger)

    def parameters(self, recurse=True):
        parameters = list(super().parameters())
        # TODO get discriminator parameters out
        for param in parameters:
            yield param

    def discriminator_parameters(self, recurse=True):
        return self._discriminator.parameters(recurse)
