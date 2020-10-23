#!/bin/python3.8
from typing import Iterator, Union

import torch
import torch.nn as nn
import numpy as np

from src.ai.architectures.bc_actor_critic_stochastic_continuous import Net as BaseNet
from src.ai.base_net import ArchitectureConfig
from src.ai.utils import mlp_creator
from src.core.data_types import Action
from src.core.logger import get_logger, cprint, MessageType
from src.core.utils import get_filename_without_extension

"""
continuous actor critic with 4d actions (X,Y) for 2 agents
"""


class Net(BaseNet):

    def __init__(self, config: ArchitectureConfig, quiet: bool = False):
        super().__init__(config=config, quiet=True)
        self.input_size = (4,)
        self.output_size = (2,)
        self.action_min = -1
        self.action_max = 1

        self._actor = mlp_creator(sizes=[self.input_size[0], 10, 2],
                                  activation=nn.Tanh(),
                                  output_activation=None)

        self._critic = mlp_creator(sizes=[self.input_size[0], 10, 1],
                                   activation=nn.Tanh(),
                                   output_activation=None)

        log_std = self._config.log_std if self._config.log_std != 'default' else -0.5
        self.log_std = torch.nn.Parameter(torch.ones(self.output_size, dtype=torch.float32) * log_std,
                                          requires_grad=True)
        if not quiet:
            self._logger = get_logger(name=get_filename_without_extension(__file__),
                                      output_path=config.output_path,
                                      quiet=False)

            cprint(f'Started.', self._logger)

            self.initialize_architecture()

    def get_action(self, inputs, train: bool = False) -> Action:
        inputs = self.process_inputs(inputs)
        output = self.sample(inputs, train=train)
        output = output.clamp(min=self.action_min, max=self.action_max)
        return Action(actor_name=get_filename_without_extension(__file__),  # assume output [1, 2] so no batch!
                      value=np.stack([*output.data.cpu().numpy().squeeze(), 0, 0], axis=-1))
