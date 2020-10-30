#!/bin/python3.8
from typing import Iterator, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

from src.ai.architectures.bc_actor_critic_stochastic_continuous import Net as BaseNet
from src.ai.base_net import ArchitectureConfig
from src.ai.utils import mlp_creator
from src.core.data_types import Action
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension


class Net(BaseNet):

    def __init__(self, config: ArchitectureConfig, quiet: bool = False):
        raise NotImplementedError('Currently this should not work as the actions recorded are 6d '
                                  'but this network only return 1d')
        super().__init__(config=config, quiet=True)
        self.input_size = (30,)
        self.output_size = (1,)
        self.action_min = -1
        self.action_max = +1
        self._actor = mlp_creator(sizes=[self.input_size[0], 64, 64, self.output_size[0]],
                                  activation=nn.Tanh(),
                                  output_activation=nn.Tanh())

        self._critic = mlp_creator(sizes=[self.input_size[0], 64, 64, 1],
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
        inputs = self.process_inputs(inputs)  # added line 15/10/2020
        output = self.sample(inputs, train=train)
        output = output.clamp(min=self.action_min, max=self.action_max)
        return Action(actor_name=get_filename_without_extension(__file__),
                      value=np.asarray([0.2, 0, 0, 0, 0, output]))
