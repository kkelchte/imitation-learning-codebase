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

        self.input_size = (1, 64, 64)
        self.input_scope = 'default'
        self.output_size = (64, 64)
        self.discrete = False
        self.dropout = nn.Dropout(p=config.dropout) if isinstance(config.dropout, float) else None
        self._config.batch_normalisation = config.batch_normalisation if isinstance(config.batch_normalisation, bool) \
            else False
        self.h = self._config.latent_dim if isinstance(config.latent_dim, int) else 32
        self.vae = self._config.vae if isinstance(config.vae, bool) else False

        if not quiet:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, self.h * (2 if self.vae else 1), 1, stride=1),
                *[nn.BatchNorm2d(self.h * (2 if self.vae else 1)), nn.LeakyReLU()] if self._config.batch_normalisation
                else [nn.LeakyReLU()],
            )
            self.decoder = nn.Sequential(nn.Conv2d(self.h, 1, 1, stride=1))
            self.initialize_architecture()
            cprint(f'Started.', self._logger)

    def _encode(self, inputs):
        """
        preprocess inputs, encode with no gradients in finetune mode, apply dropout if necessary
        :param inputs: numpy array, torch tensor, list, ...
        :param train: bool setting training mode on / off in super class
        :return:
        """
        inputs = self.process_inputs(inputs=inputs)
        if self._config.finetune:
            with torch.no_grad():
                x = self.encoder(inputs)
        else:
            x = self.encoder(inputs)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

    def _decode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(inputs).squeeze(dim=1)

    def _forward_with_distribution(self, inputs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        return network outputs and latent distribution
        """
        x = self._encode(inputs)
        mean = x[:, :self.h]
        std = torch.exp(x[:, self.h:]/2)
        x = mean + std * torch.rand_like(std)
        x = self._decode(x)
        return x, mean, std

    def forward(self, inputs, train: bool = False) -> torch.Tensor:
        """
        Outputs steering action only
        """
        self.set_mode(train)
        if not self.vae:
            x = self._encode(inputs)
            x = self._decode(x)
        else:
            x, _, _ = self.forward_with_distribution(inputs)
        return x

    def get_action(self, inputs, train: bool = False) -> Action:
        raise NotImplementedError
