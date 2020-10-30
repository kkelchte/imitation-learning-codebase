#!/bin/python3.8
from typing import Tuple

import torch
import torch.nn as nn

from src.ai.architectural_components import ResidualBlock
from src.ai.base_net import BaseNet, ArchitectureConfig
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
        self._logger = get_logger(name=get_filename_without_extension(__file__),
                                  output_path=config.output_path,
                                  quiet=False)

        self.input_size = (1, 200, 200)
        self.input_scope = 'default'
        self.output_size = (200, 200)
        self.discrete = False
        self._config.batch_normalisation = config.batch_normalisation if isinstance(config.batch_normalisation, bool) \
            else False

        if not quiet:
            self.sigmoid = nn.Sigmoid()
            self.residual_1 = ResidualBlock(input_channels=1,
                                            output_channels=32,
                                            batch_norm=self._config.batch_normalisation,
                                            activation=torch.nn.ReLU(),
                                            strides=(1, 1),
                                            padding=(1, 1),
                                            kernel_sizes=(3, 3))
            self.side_logit_1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
            self.weight_1 = nn.Parameter(torch.as_tensor(1/4), requires_grad=True)

            self.residual_2 = ResidualBlock(input_channels=32,
                                            output_channels=32,
                                            batch_norm=self._config.batch_normalisation,
                                            activation=torch.nn.ReLU(),
                                            strides=(2, 1),
                                            padding=(1, 1),
                                            kernel_sizes=(3, 3))
            self.side_logit_2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
            self.weight_2 = nn.Parameter(torch.as_tensor(1/4), requires_grad=True)
            self.upsample_2 = nn.Upsample(scale_factor=2, mode='nearest')

            self.residual_3 = ResidualBlock(input_channels=32,
                                            output_channels=32,
                                            batch_norm=self._config.batch_normalisation,
                                            activation=torch.nn.ReLU(),
                                            strides=(2, 1),
                                            padding=(1, 1),
                                            kernel_sizes=(3, 3))
            self.side_logit_3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
            self.weight_3 = nn.Parameter(torch.as_tensor(1/4), requires_grad=True)
            self.upsample_3 = nn.Upsample(scale_factor=4, mode='nearest')

            self.residual_4 = ResidualBlock(input_channels=32,
                                            output_channels=32,
                                            batch_norm=self._config.batch_normalisation,
                                            activation=torch.nn.ReLU(),
                                            strides=(2, 1),
                                            padding=(1, 1),
                                            kernel_sizes=(3, 3))
            self.side_logit_4 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
            self.weight_4 = nn.Parameter(torch.as_tensor(1/4), requires_grad=True)
            self.upsample_4 = nn.Upsample(scale_factor=8, mode='nearest')

            self.initialize_architecture()
            cprint(f'Started.', self._logger)

    def forward_with_all_outputs(self, inputs, train: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                                             torch.Tensor, torch.Tensor]:
        self.set_mode(train)
        processed_inputs = self.process_inputs(inputs)
        x1 = self.residual_1(processed_inputs)
        out1 = self.side_logit_1(x1)
        prob1 = self.sigmoid(out1).squeeze(dim=1)

        x2 = self.residual_2(x1)
        out2 = self.side_logit_2(x2)
        prob2 = self.upsample_2(self.sigmoid(out2)).squeeze(dim=1)

        x3 = self.residual_3(x2)
        out3 = self.side_logit_3(x3)
        prob3 = self.upsample_3(self.sigmoid(out3)).squeeze(dim=1)

        x4 = self.residual_4(x3)
        out4 = self.side_logit_4(x4)
        prob4 = self.upsample_4(self.sigmoid(out4)).squeeze(dim=1)

        final_logit = self.weight_1 * out1 + \
            self.weight_2 * self.upsample_2(out2) + \
            self.weight_3 * self.upsample_3(out3) + \
            self.weight_4 * self.upsample_4(out4)
        final_prob = self.sigmoid(final_logit).squeeze(dim=1)
        return prob1, prob2, prob3, prob4, final_prob

    def forward(self, inputs, train: bool = False) -> torch.Tensor:
        _, _, _, _, final_prob = self.forward_with_all_outputs(inputs, train)
        return final_prob

    def get_action(self, inputs, train: bool = False) -> Action:
        raise NotImplementedError


class ImageNet(Net):

    def __init__(self, config: ArchitectureConfig, quiet: bool = False):
        super().__init__(config=config, quiet=False)
        self._imagenet_output = torch.nn.Linear(32*25*25, 1000)

    def forward(self, inputs, train: bool = False) -> torch.Tensor:
        self.set_mode(train)
        processed_inputs = self.process_inputs(inputs)
        x1 = self.residual_1(processed_inputs)
        x2 = self.residual_2(x1)
        x3 = self.residual_3(x2)
        x4 = self.residual_4(x3)
        return self._imagenet_output(x4.flatten(start_dim=1))


