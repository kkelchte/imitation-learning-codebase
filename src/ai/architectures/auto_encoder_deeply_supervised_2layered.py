#!/bin/python3.8
from typing import Tuple

import torch
import torch.nn as nn

from src.ai.architectural_components import ResidualBlock
from src.ai.base_net import BaseNet, ArchitectureConfig
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
        self._logger = get_logger(name=get_filename_without_extension(__file__),
                                  output_path=config.output_path,
                                  quiet=False)

        self.input_size = (1, 200, 200)
        self.input_scope = 'default'
        self.output_size = (200, 200)
        self.discrete = False
        self.dropout = nn.Dropout(p=config.dropout) if isinstance(config.dropout, float) else None
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

            self.initialize_architecture()
            cprint(f'Started.', self._logger)

    def forward_with_all_outputs(self, inputs, train: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        processed_inputs = self.process_inputs(inputs, train)
        x1 = self.residual_1(processed_inputs)
        out1 = self.side_logit_1(x1)
        prob1 = self.sigmoid(out1).squeeze(dim=1)

        x2 = self.residual_2(x1)
        out2 = self.side_logit_2(x2)
        prob2 = self.upsample_2(self.sigmoid(out2)).squeeze(dim=1)

        final_logit = self.weight_1 * out1 + self.weight_2 * self.upsample_2(out2)
        final_prob = self.sigmoid(final_logit).squeeze(dim=1)
        return prob1, prob2, final_prob

    def forward(self, inputs, train: bool = False) -> torch.Tensor:
        _, _, final_prob = self.forward_with_all_outputs(inputs, train)
        return final_prob

    def get_action(self, inputs, train: bool = False) -> Action:
        raise NotImplementedError
