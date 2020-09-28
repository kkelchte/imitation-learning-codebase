#!/bin/python3.8

import torch.nn as nn

from src.ai.base_net import ArchitectureConfig
from src.ai.architectures.auto_encoder_conv1_200 import Net as BaseNet
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension

"""
Four encoding and four decoding layers with dropout.
Expects 3x200x200 inputs and outputs 200x200
"""


class Net(BaseNet):

    def __init__(self, config: ArchitectureConfig, quiet: bool = False):
        super().__init__(config=config, quiet=True)
        self._logger = get_logger(name=get_filename_without_extension(__file__),
                                  output_path=config.output_path,
                                  quiet=False)
        if not quiet:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, self.h, 3, stride=1, padding=1),
                *[nn.BatchNorm2d(self.h), nn.LeakyReLU()] if self._config.batch_normalisation
                else [nn.LeakyReLU()],
                nn.Conv2d(self.h, self.h * (2 if self.vae else 1), 3, stride=1, padding=1),
                *[nn.BatchNorm2d(self.h * (2 if self.vae else 1)), nn.LeakyReLU()] if self._config.batch_normalisation
                else [nn.LeakyReLU()],
            )
            self.decoder = nn.Sequential(
                nn.Conv2d(self.h, self.h, 3, stride=1, padding=1),
                nn.Conv2d(self.h, 1, 3, stride=1, padding=1),
            )
            self.initialize_architecture()
            cprint(f'Started.', self._logger)
