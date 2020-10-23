#!/bin/python3.8
import torch
import torch.nn as nn

from src.ai.architectures.bc_actor_critic_stochastic_continuous import Net as BaseNet
from src.ai.base_net import ArchitectureConfig
from src.ai.utils import mlp_creator
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension

"""
Humanoid: observation 376, action 17 continuous, stochastic actor-critic
"""


class Net(BaseNet):

    def __init__(self, config: ArchitectureConfig, quiet: bool = False):
        super().__init__(config=config, quiet=True)

        self.input_size = (376,)
        self.output_size = (17,)
        self.action_min = -0.4
        self.action_max = 0.4
        log_std = self._config.log_std if self._config.log_std != 'default' else -0.5
        self.log_std = torch.nn.Parameter(torch.ones(self.output_size, dtype=torch.float32) * log_std,
                                          requires_grad=True)

        self._actor = mlp_creator(sizes=[self.input_size[0], 64, 64, self.output_size[0]],
                                  activation=nn.Tanh(),
                                  output_activation=None)

        self._critic = mlp_creator(sizes=[self.input_size[0], 64, 64, 1],
                                   activation=nn.Tanh(),
                                   output_activation=None)

        if not quiet:
            self._logger = get_logger(name=get_filename_without_extension(__file__),
                                      output_path=config.output_path,
                                      quiet=False)

            cprint(f'Started.', self._logger)
            self.initialize_architecture()
