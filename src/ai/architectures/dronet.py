#!/bin/python3.8
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ai.base_net import BaseNet, ArchitectureConfig
from src.ai.utils import mlp_creator
from src.core.data_types import Action
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension

"""
Tiny four encoding and three decoding layers with dropout.
Expects 3x128x128 inputs and outputs 1c 
"""


class Net(BaseNet):

    def __init__(self, config: ArchitectureConfig, quiet: bool = False):
        super().__init__(config=config, quiet=True)
        self._logger = get_logger(name=get_filename_without_extension(__file__),
                                  output_path=config.output_path,
                                  quiet=False)
        if not quiet:
            cprint(f'Started.', self._logger)

        self.input_size = (1, 200, 200)
        self.output_size = (6,)
        self.discrete = False
        self.dropout = nn.Dropout(p=config.dropout) if config.dropout != 'default' else None

        self.conv2d_1 = nn.Conv2d(in_channels=self.input_size[0], out_channels=32,
                                  kernel_size=5, stride=2, padding=1, bias=True)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # First residual block
        self.batch_normalization_1 = nn.BatchNorm2d(32)
        self.conv2d_2 = nn.Conv2d(in_channels=32, out_channels=32,
                                  kernel_size=3, stride=2, padding=1, bias=True)
        self.batch_normalization_2 = nn.BatchNorm2d(32)
        self.conv2d_3 = nn.Conv2d(in_channels=32, out_channels=32,
                                  kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2d_4 = nn.Conv2d(in_channels=32, out_channels=32,
                                  kernel_size=1, stride=2, padding=0, bias=True)
        # Second residual block
        self.batch_normalization_3 = nn.BatchNorm2d(32)
        self.conv2d_5 = nn.Conv2d(in_channels=32, out_channels=64,
                                  kernel_size=3, stride=2, padding=1, bias=True)
        self.batch_normalization_4 = nn.BatchNorm2d(64)
        self.conv2d_6 = nn.Conv2d(in_channels=64, out_channels=64,
                                  kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2d_7 = nn.Conv2d(in_channels=32, out_channels=64,
                                  kernel_size=1, stride=2, padding=0, bias=True)
        # Third residual block
        self.batch_normalization_5 = nn.BatchNorm2d(64)
        self.conv2d_8 = nn.Conv2d(in_channels=64, out_channels=128,
                                  kernel_size=3, stride=2, padding=1, bias=True)
        self.batch_normalization_6 = nn.BatchNorm2d(128)
        self.conv2d_9 = nn.Conv2d(in_channels=128, out_channels=128,
                                  kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2d_10 = nn.Conv2d(in_channels=64, out_channels=128,
                                   kernel_size=1, stride=2, padding=0, bias=True)

        #self.dense_1 = nn.Linear(6272, 1)
        #self.dense_2 = nn.Linear(6272, 1)

        self.decoder = mlp_creator(sizes=[6272, 2056, self.output_size[0]],
                                   activation=nn.ReLU(),
                                   output_activation=nn.Tanh(),
                                   bias_in_last_layer=False)

    def feature_extract(self, inputs: torch.Tensor) -> torch.Tensor:
        verbose = False
        x1 = self.maxpool_1(self.conv2d_1(inputs))
        if verbose:
            print(f'x1: {x1.size()}')
        # first residual block
        x2 = F.relu(self.batch_normalization_1(x1))
        x2 = F.relu(self.batch_normalization_2(self.conv2d_2(x2)))
        if verbose:
            print(f'x2: {x2.size()}')
        x2 = self.conv2d_3(x2)
        if verbose:
            print(f'x2: {x2.size()}')
        out_res = self.conv2d_4(x1)
        if verbose:
            print(f'out_res: {out_res.size()}')
        x2 += out_res
        if verbose:
            print(f'x2: {x2.size()}')

        # second residual block
        x3 = F.relu(self.batch_normalization_3(x2))
        if verbose:
            print(f'x3: {x3.size()}')
        x3 = F.relu(self.batch_normalization_4(self.conv2d_5(x3)))
        if verbose:
            print(f'x3: {x3.size()}')
        x3 = self.conv2d_6(x3)
        if verbose:
            print(f'x3: {x3.size()}')
        out_res = self.conv2d_7(x2)
        if verbose:
            print(f'out_res: {out_res.size()}')
        x3 += out_res
        if verbose:
            print(f'x3: {x3.size()}')
        # third residual block
        x4 = F.relu(self.batch_normalization_5(x3))
        if verbose:
            print(f'x4: {x4.size()}')
        x4 = F.relu(self.batch_normalization_6(self.conv2d_8(x4)))
        if verbose:
            print(f'x4: {x4.size()}')
        x4 = self.conv2d_9(x4)
        if verbose:
            print(f'x4: {x4.size()}')
        out_res = self.conv2d_10(x3)
        if verbose:
            print(f'out_res: {out_res.size()}')
        x4 += out_res
        if verbose:
            print(f'x4: {x4.size()}')
        # flatten
        return x4.view(x4.size(0), -1)

    def forward(self, inputs, train: bool = False) -> torch.Tensor:
        """
        Outputs steering action only
        """
        self.set_mode(train)
        inputs = self.process_inputs(inputs=inputs)
        if self._config.finetune:
            with torch.no_grad():
                x4 = self.feature_extract(inputs)
        else:
            x4 = self.feature_extract(inputs)
        if self.dropout is not None:
            x4 = self.dropout(x4)
        return self.decoder(x4)

    def get_action(self, inputs, train: bool = False) -> Action:
        output = self.forward(inputs, train=train)
        return Action(actor_name=get_filename_without_extension(__file__),
                      value=output.data)

