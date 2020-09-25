#!/bin/python3.8
from typing import Iterator

import torch
import torch.nn as nn
from torch.distributions import Normal

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
        self.action_min = -2
        self.action_max = 2
        self.discrete = False

        log_std = self._config.log_std if self._config.log_std != 'default' else 0.1
        self.log_std = torch.nn.Parameter(torch.as_tensor([log_std] * self.output_size[0]), requires_grad=False)

        self._actor = mlp_creator(sizes=[self.input_size[0], 64, 64, self.output_size[0]],
                                  activation=nn.Tanh(),
                                  output_activation=None)

        self._critic = mlp_creator(sizes=[self.input_size[0], 64, 64, 1],
                                   activation=nn.Tanh(),
                                   output_activation=None)
        self.initialize_architecture()

    def get_actor_parameters(self) -> Iterator:
        return self._actor.parameters()

    def get_critic_parameters(self) -> Iterator:
        return self._critic.parameters()

    def _policy_distribution(self, inputs: torch.Tensor, train: bool = True) -> Normal:
        inputs = self.process_inputs(inputs=inputs, train=train)
        logits = self.action_max * self._actor(inputs)
        return Normal(logits, torch.exp(self.log_std))

    def get_policy_entropy(self, inputs: torch.Tensor, train: bool = True) -> torch.Tensor:
        distribution = self._policy_distribution(inputs=inputs, train=train)
        return distribution.entropy().sum(dim=1)

    def get_action(self, inputs, train: bool = False) -> Action:
        output = self._policy_distribution(inputs, train).sample()
        output = output.clamp(min=self.action_min, max=self.action_max)
        return Action(actor_name=get_filename_without_extension(__file__),
                      value=output)

    def policy_log_probabilities(self, inputs, actions, train: bool = True) -> torch.Tensor:
        inputs = self.process_inputs(inputs=inputs, train=train)
        actions = self.process_inputs(inputs=actions, train=train)  # preprocess list of Actions
        return self._policy_distribution(inputs).log_prob(actions).sum(dim=-1)

    def critic(self, inputs, train: bool = False) -> torch.Tensor:
        inputs = self.process_inputs(inputs=inputs, train=train)
        return self._critic(inputs)
