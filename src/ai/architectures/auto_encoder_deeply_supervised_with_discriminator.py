#!/bin/python3.8
from typing import Tuple

import torch
import torch.nn as nn

from src.ai.architectural_components import ResidualBlock
from src.ai.architectures.bc_deeply_supervised_auto_encoder import Net as BaseNet
from src.ai.base_net import ArchitectureConfig
from src.ai.utils import mlp_creator, conv_creator
from src.core.data_types import Action
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension

"""
Deep Supervision net with discriminator.
Discriminator is used to improve the predictions from the network on unlabeled real data.
Discriminator discriminates between simulated (training) data prediction (0) and real (test) data prediction (1).
The main network can then be trained also on unlabeled real data to minimize the discriminators output.
"""


class Net(BaseNet):

    def __init__(self, config: ArchitectureConfig, quiet: bool = False):
        super().__init__(config=config, quiet=True)
        self._deeply_supervised_parameter_names = [name for name, _ in self.named_parameters()]
        self._discriminator = conv_creator(channels=[1, 3, 6, 9],
                                           kernel_sizes=[5, 5, 5],
                                           strides=[3, 3, 3],
                                           activation=nn.LeakyReLU(),
                                           output_activation=nn.LeakyReLU(),
                                           batch_norm=self._config.batch_normalisation)
        self._discriminator_decision = mlp_creator([9*6*6, 1], output_activation=nn.Sigmoid(),
                                                   bias_in_last_layer=False)
        if not quiet:
            self._logger = get_logger(name=get_filename_without_extension(__file__),
                                      output_path=config.output_path,
                                      quiet=False)
            self.initialize_architecture()
            cprint(f'Started.', self._logger)

    def deeply_supervised_parameters(self, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            if name in self._deeply_supervised_parameter_names:
                yield param

    def discriminator_parameters(self, recurse=True):
        for p in self._discriminator.parameters(recurse=recurse):
            yield p
        for p in self._discriminator_decision.parameters(recurse=recurse):
            yield p

    def forward_with_all_outputs(self, inputs, train: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                                             torch.Tensor, torch.Tensor]:
        for p in self.deeply_supervised_parameters():
            p.requires_grad = train
        return super().forward_with_all_outputs(inputs, train=train)

    def discriminate(self, predictions, train: bool = False) -> torch.Tensor:
        """
        Evaluate predictions on whether they come from simulated (0) or real (1) data
        :param predictions: NxCxHxW with CxHxW corresponding to the output size
        :param train: train the discriminator part or evaluate
        :return: output 0 --> simulated, 1 --> real
        """
        self._discriminator.train(train)
        for p in self.discriminator_parameters():
            p.requires_grad = train
        feature = self._discriminator(predictions).view(-1, 9*6*6)
        return self._discriminator_decision(feature)
