#!/bin/python3.8
from collections import OrderedDict
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

            self.conv0 = torch.nn.Conv2d(in_channels=1,
                                         out_channels=32,
                                         kernel_size=3,
                                         padding=1,
                                         stride=1)
            self.conv1_1 = torch.nn.Conv2d(in_channels=32,
                                           out_channels=32,
                                           kernel_size=3,
                                           padding=1,
                                           stride=1)
            self.conv1_2 = torch.nn.Conv2d(in_channels=32,
                                           out_channels=32,
                                           kernel_size=3,
                                           padding=1,
                                           stride=1)
            self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

            self.side_logit_1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
            self.weight_1 = nn.Parameter(torch.as_tensor(1/4), requires_grad=True)

            self.conv2_1 = torch.nn.Conv2d(in_channels=32,
                                           out_channels=32,
                                           kernel_size=3,
                                           padding=1,
                                           stride=1)
            self.conv2_2 = torch.nn.Conv2d(in_channels=32,
                                           out_channels=32,
                                           kernel_size=3,
                                           padding=1,
                                           stride=1)
            self.conv2_1.weight = self.conv1_1.weight
            self.conv2_2.weight = self.conv1_2.weight
            self.side_logit_2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
            self.weight_2 = nn.Parameter(torch.as_tensor(1/4), requires_grad=True)
            self.upsample_2 = nn.Upsample(scale_factor=2, mode='nearest')

            self.conv3_1 = torch.nn.Conv2d(in_channels=32,
                                           out_channels=32,
                                           kernel_size=3,
                                           padding=1,
                                           stride=1)
            self.conv3_2 = torch.nn.Conv2d(in_channels=32,
                                           out_channels=32,
                                           kernel_size=3,
                                           padding=1,
                                           stride=1)
            self.conv3_1.weight = self.conv1_1.weight
            self.conv3_2.weight = self.conv1_2.weight
            self.side_logit_3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
            self.weight_3 = nn.Parameter(torch.as_tensor(1/4), requires_grad=True)
            self.upsample_3 = nn.Upsample(scale_factor=4, mode='nearest')

            self.conv4_1 = torch.nn.Conv2d(in_channels=32,
                                           out_channels=32,
                                           kernel_size=3,
                                           padding=1,
                                           stride=1)
            self.conv4_2 = torch.nn.Conv2d(in_channels=32,
                                           out_channels=32,
                                           kernel_size=3,
                                           padding=1,
                                           stride=1)
            self.conv4_1.weight = self.conv1_1.weight
            self.conv4_2.weight = self.conv1_2.weight
            self.side_logit_4 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
            self.weight_4 = nn.Parameter(torch.as_tensor(1/4), requires_grad=True)
            self.upsample_4 = nn.Upsample(scale_factor=8, mode='nearest')

            self.initialize_architecture()
            cprint(f'Started.', self._logger)

    def forward_with_all_outputs(self, inputs, train: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                                             torch.Tensor, torch.Tensor]:
        processed_inputs = self.process_inputs(inputs, train)
        # bring 1 channel to 32 channels
        x0 = self.conv0(processed_inputs)
        # apply first filterset at 32x200x200
        x1 = self.conv1_1(x0).relu()
        x1 = x0 + self.conv1_2(x1).relu()
        # extract primary output for deep supervision
        out1 = self.side_logit_1(x1)
        prob1 = self.sigmoid(out1).squeeze(dim=1)

        # downscale with max pool and reuse conv1 and conv2
        x1 = self.maxpool(x1)
        # apply first filterset at 32x100x100
        x2 = self.conv2_1(x1).relu()
        x2 = x1 + self.conv2_2(x2).relu()
        # extract secondary output for deep supervision
        out2 = self.side_logit_2(x2)
        prob2 = self.upsample_2(self.sigmoid(out2)).squeeze(dim=1)

        # downscale with max pool and reuse conv1 and conv2
        x2 = self.maxpool(x2)
        # apply first filterset at 32x100x100
        x3 = self.conv3_1(x2).relu()
        x3 = x2 + self.conv3_2(x3).relu()
        # extract secondary output for deep supervision
        out3 = self.side_logit_3(x3)
        prob3 = self.upsample_3(self.sigmoid(out3)).squeeze(dim=1)

        # downscale with max pool and reuse conv1 and conv2
        x3 = self.maxpool(x3)
        # apply first filterset at 32x100x100
        x4 = self.conv4_1(x3).relu()
        x4 = x3 + self.conv4_2(x4).relu()
        # extract secondary output for deep supervision
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
        processed_inputs = self.process_inputs(inputs, train)
        x1 = self.residual_1(processed_inputs)
        x2 = self.residual_2(x1)
        x3 = self.residual_3(x2)
        x4 = self.residual_4(x3)
        return self._imagenet_output(x4.flatten(start_dim=1))


