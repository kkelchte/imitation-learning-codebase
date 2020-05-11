#!/bin/python3.7
from typing import Iterator

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.categorical import Categorical

from src.ai.base_net import BaseNet, ArchitectureConfig
from src.ai.utils import mlp_creator
from src.core.data_types import Action
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension

"""
Pendulum-v0	action space: continuous 1[-2.0 : 2.0]	observation space: Box(3,)
"""


class Net(BaseNet):

    def __init__(self, config: ArchitectureConfig, quiet: bool = False):
        super().__init__(config=config)
        if not quiet:
            self._logger = get_logger(name=get_filename_without_extension(__file__),
                                      output_path=config.output_path,
                                      quiet=False)

            cprint(f'Started.', self._logger)

        self.input_size = (3,)
        self.output_size = (1, 1)
        self.discrete = False

        log_std = self._config.log_std if self._config.log_std != 'default' else 0.5
        self.log_std = torch.nn.Parameter(torch.as_tensor([log_std] * self.output_size[0]), requires_grad=False)

        self._actor = mlp_creator(sizes=[self.input_size[0], 25, 25, self.output_size[0]],
                                  activation=nn.ReLU,
                                  output_activation=nn.Tanh)

        self._critic = mlp_creator(sizes=[self.input_size[0], 25, 25, 1],
                                   activation=nn.ReLU,
                                   output_activation=None)
        self.load_network_weights()

    def get_actor_parameters(self) -> Iterator:
        return self._actor.parameters()

    def get_critic_parameters(self) -> Iterator:
        return self._critic.parameters()

    def _policy_distribution(self, inputs: torch.Tensor, train: bool = True) -> Normal:
        inputs = super().forward(inputs=inputs, train=train)
        logits = 2 * self._actor(inputs)
        return Normal(logits, torch.exp(self.log_std))

    def get_action(self, inputs, train: bool = False) -> Action:
        output = self._policy_distribution(inputs, train).sample()
        output = output.clamp(min=-2, max=2)
        return Action(actor_name=get_filename_without_extension(__file__),
                      value=output)

    def policy_log_probabilities(self, inputs, actions, train: bool = True) -> torch.Tensor:
        inputs = super().forward(inputs=inputs, train=train)
        actions = super().forward(inputs=actions, train=train)  # preprocess list of Actions
        return self._policy_distribution(inputs).log_prob(actions)

    def critic(self, inputs, train: bool = False) -> torch.Tensor:
        inputs = super().forward(inputs=inputs, train=train)
        return self._critic(inputs)


