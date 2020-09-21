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
        self.h = 128
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 5, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 5, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(128, 2 * self.h, 5, stride=2),
        ) if not self._config.batch_normalisation else \
            nn.Sequential(
                nn.Conv2d(1, 32, 5, stride=2),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.Conv2d(32, 64, 5, stride=2),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.Conv2d(64, 128, 5, stride=2),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.Conv2d(128, 2 * self.h, 5, stride=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.h, 128, 5, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, 6, stride=2, padding=0)
        ) if not self._config.batch_normalisation else \
            nn.Sequential(
                nn.ConvTranspose2d(self.h, 128, 5, stride=2, padding=0),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(128, 64, 5, stride=2, padding=0),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(64, 32, 6, stride=2, padding=0),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(32, 1, 6, stride=2, padding=0)
            )
        self.initialize_architecture()

    def forward_with_distribution(self, inputs, train: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Outputs steering action only
        """
        inputs = super().forward(inputs=inputs, train=train)
        if self._config.finetune:
            with torch.no_grad():
                x = self.encoder(inputs)
        else:
            x = self.encoder(inputs)
        if self.dropout is not None and train:
            x = self.dropout(x)
        mean = x[:, :self.h]
        std = torch.exp(x[:, self.h:]/2)
        x = mean + std * torch.rand_like(std)
        x = self.decoder(x).squeeze(1)
        return x, mean, std

    def forward(self, inputs, train: bool = False) -> torch.Tensor:
        x, _, _ = self.forward_with_distribution(inputs)
        return x

    def get_action(self, inputs, train: bool = False) -> Action:
        output, _, _ = self.forward(inputs, train=train)
        return Action(actor_name=get_filename_without_extension(__file__),
                      value=output.data)

