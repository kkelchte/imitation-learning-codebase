#!/bin/python3.8
from typing import Iterator

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from src.ai.base_net import BaseNet, ArchitectureConfig
from src.ai.utils import mlp_creator
from src.core.data_types import Action
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension

"""
CartPole-v0	action space: discrete 2	observation space: Box(4,)
Pendulum-v0	action space: continuous 1[-2.0 : 2.0]	observation space: Box(3,)
"""


class Net(BaseNet):

    def __init__(self, config: ArchitectureConfig, quiet: bool = False):
        super().__init__(config=config, quiet=True)
        if not quiet:
            self._logger = get_logger(name=get_filename_without_extension(__file__),
                                      output_path=config.output_path,
                                      quiet=False)

            cprint(f'Started.', self._logger)

        self.input_size = (4,)
        self.output_size = (2,)
        self.discrete = True
        self._actor = mlp_creator(sizes=[self.input_size[0], 20, 20, 20, self.output_size[0]],
                                  activation=nn.ReLU,
                                  output_activation=None)

        self._critic = mlp_creator(sizes=[self.input_size[0], 20, 20, 20, 1],
                                   activation=nn.ReLU,
                                   output_activation=None)
        self.load_network_weights()

    def get_actor_parameters(self) -> Iterator:
        return self._actor.parameters()

    def get_critic_parameters(self) -> Iterator:
        return self._critic.parameters()

    def _policy_distribution(self, inputs: torch.Tensor, train: bool = True) -> Categorical:
        inputs = super().forward(inputs=inputs, train=train)
        logits = self._actor(inputs)
        return Categorical(logits=logits)

    def get_action(self, inputs, train: bool = False) -> Action:
        output = self._policy_distribution(inputs, train).sample()
        return Action(actor_name=get_filename_without_extension(__file__),
                      value=output.item())

    def policy_log_probabilities(self, inputs, actions, train: bool = True) -> torch.Tensor:
        actions = super().forward(inputs=actions, train=train)
        return self._policy_distribution(inputs, train=train).log_prob(actions)

    def critic(self, inputs, train: bool = False) -> torch.Tensor:
        inputs = super().forward(inputs=inputs, train=train)
        return self._critic(inputs)
