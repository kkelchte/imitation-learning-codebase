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

        self.input_size = (1, 200, 200)
        self.input_scope = 'default'
        self.output_size = (200, 200)
        self.discrete = False
        self._config.batch_normalisation = config.batch_normalisation if isinstance(config.batch_normalisation, bool) \
            else False
        self.sigmoid = nn.Sigmoid()
        self.pool_features = torch.nn.MaxPool2d(5)
        self.conv0 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=32,
                                     kernel_size=3,
                                     padding=1,
                                     stride=1)
        self.residual_1 = ResidualBlock(input_channels=32,
                                        output_channels=32,
                                        batch_norm=self._config.batch_normalisation,
                                        activation=torch.nn.ReLU(),
                                        strides=(1, 1),
                                        padding=(1, 1),
                                        kernel_sizes=(3, 3))
        self.side_logit_1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.weight_1 = nn.Parameter(torch.as_tensor(1 / 4), requires_grad=True)

        self.residual_2 = ResidualBlock(input_channels=32,
                                        output_channels=32,
                                        batch_norm=self._config.batch_normalisation,
                                        activation=torch.nn.ReLU(),
                                        strides=(2, 1),
                                        padding=(1, 1),
                                        kernel_sizes=(3, 3))
        self.side_logit_2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.weight_2 = nn.Parameter(torch.as_tensor(1 / 4), requires_grad=True)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.residual_3 = ResidualBlock(input_channels=32,
                                        output_channels=32,
                                        batch_norm=self._config.batch_normalisation,
                                        activation=torch.nn.ReLU(),
                                        strides=(2, 1),
                                        padding=(1, 1),
                                        kernel_sizes=(3, 3))
        self.side_logit_3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.weight_3 = nn.Parameter(torch.as_tensor(1 / 4), requires_grad=True)
        self.upsample_3 = nn.Upsample(scale_factor=4, mode='nearest')

        self.residual_4 = ResidualBlock(input_channels=32,
                                        output_channels=32,
                                        batch_norm=self._config.batch_normalisation,
                                        activation=torch.nn.ReLU(),
                                        strides=(2, 1),
                                        padding=(1, 1),
                                        kernel_sizes=(3, 3))
        self.side_logit_4 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.weight_4 = nn.Parameter(torch.as_tensor(1 / 4), requires_grad=True)
        self.upsample_4 = nn.Upsample(scale_factor=8, mode='nearest')
        if not quiet:
            self._logger = get_logger(name=get_filename_without_extension(__file__),
                                      output_path=config.output_path,
                                      quiet=False)

            self.initialize_architecture()
            cprint(f'Started.', self._logger)

    def forward_with_intermediate_outputs(self, inputs, train: bool = False) -> dict:
        self.set_mode(train)
        processed_inputs = self.process_inputs(inputs)

        results = {'x1': self.residual_1(self.conv0(processed_inputs))}
        results['out1'] = self.side_logit_1(results['x1'])
        results['prob1'] = self.sigmoid(results['out1']).squeeze(dim=1)

        results['x2'] = self.residual_2(results['x1'])
        results['out2'] = self.side_logit_2(results['x2'])
        results['prob2'] = self.upsample_2(self.sigmoid(results['out2'])).squeeze(dim=1)

        results['x3'] = self.residual_3(results['x2'])
        results['out3'] = self.side_logit_3(results['x3'])
        results['prob3'] = self.upsample_3(self.sigmoid(results['out3'])).squeeze(dim=1)

        results['x4'] = self.residual_4(results['x3'])
        results['out4'] = self.side_logit_4(results['x4'])
        results['prob4'] = self.upsample_4(self.sigmoid(results['out4'])).squeeze(dim=1)

        final_logit = self.weight_1 * results['out1'] + \
                      self.weight_2 * self.upsample_2(results['out2']) + \
                      self.weight_3 * self.upsample_3(results['out3']) + \
                      self.weight_4 * self.upsample_4(results['out4'])
        results['final_prob'] = self.sigmoid(final_logit).squeeze(dim=1)
        return results

    def forward_with_all_outputs(self, inputs, train: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                                             torch.Tensor, torch.Tensor]:
        results = self.forward_with_intermediate_outputs(inputs)
        return results['prob1'], results['prob2'], results['prob3'], results['prob4'], results['final_prob']

    def forward(self, inputs, train: bool = False) -> torch.Tensor:
        _, _, _, _, final_prob = self.forward_with_all_outputs(inputs, train)
        return final_prob

    def get_action(self, inputs, train: bool = False) -> Action:
        raise NotImplementedError

    def get_features(self, inputs, train: bool = False) -> torch.Tensor:
        results = self.forward_with_intermediate_outputs(inputs, train=train)
        pooled_features = self.pool_features(results['x4'])
        #return torch.cat([v.flatten(start_dim=1, end_dim=3)
        #                  for v in [results['x1'], results['x2'], results['x3'], results['x4']]], dim=1)
        return pooled_features


class ImageNet(Net):

    def __init__(self, config: ArchitectureConfig, quiet: bool = False):
        super().__init__(config=config, quiet=False)
        self._imagenet_output = torch.nn.Linear(32 * 25 * 25, 1000)

    def forward(self, inputs, train: bool = False) -> torch.Tensor:
        self.set_mode(train)
        processed_inputs = self.process_inputs(inputs)
        x1 = self.residual_1(processed_inputs)
        x2 = self.residual_2(x1)
        x3 = self.residual_3(x2)
        x4 = self.residual_4(x3)
        return self._imagenet_output(x4.flatten(start_dim=1))
