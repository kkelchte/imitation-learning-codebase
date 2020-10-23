#!/bin/python3.8
import torch.nn as nn

from src.ai.base_net import ArchitectureConfig
from src.ai.architectures.bc_actor_critic_stochastic_discrete import Net as BaseNet
from src.ai.utils import mlp_creator
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension

"""
CartPole-v0	action space: discrete 2	observation space: Box(4,)
Pendulum-v0	action space: continuous 1[-2.0 : 2.0]	observation space: Box(3,)
"""


class Net(BaseNet):

    def __init__(self, config: ArchitectureConfig, quiet: bool = False):
        super().__init__(config=config, quiet=True)
        self.input_size = (4,)
        self.output_size = (2,)
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
