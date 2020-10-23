#!/bin/python3.8
import torch.nn as nn

from src.ai.architectures.bc_variational_auto_encoder import Net as BaseNet
from src.ai.base_net import ArchitectureConfig
from src.ai.utils import mlp_creator
from src.core.data_types import Action
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension

"""
One encoding and one decoding layers with dropout.
Expects 1x64x64 inputs and outputs 64x64
"""


class Net(BaseNet):

    def __init__(self, config: ArchitectureConfig, quiet: bool = False):
        super().__init__(config=config, quiet=True)
        self.input_size = (1, 64, 64)
        self.output_size = (64, 64)

        if not quiet:
            self._logger = get_logger(name=get_filename_without_extension(__file__),
                                      output_path=config.output_path,
                                      quiet=False)

            self.encoder = nn.Sequential(
                nn.Conv2d(1, self.h * (2 if self.vae else 1), 1, stride=1),
                *[nn.BatchNorm2d(self.h * (2 if self.vae else 1)), nn.LeakyReLU()] if self._config.batch_normalisation
                else [nn.LeakyReLU()],
            )
            self.decoder = nn.Sequential(nn.Conv2d(self.h, 1, 1, stride=1))
            self.initialize_architecture()
            cprint(f'Started.', self._logger)
