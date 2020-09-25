#!/bin/python3.8
from typing import Iterator, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

from src.ai.base_net import BaseNet, ArchitectureConfig
from src.ai.utils import mlp_creator, initialize_weights
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

        self.input_size = (376,)
        self.output_size = (17,)
        self.action_min = -0.4
        self.action_max = 0.4

        self.discrete = False

        self._actor = mlp_creator(sizes=[self.input_size[0], 64, 64, self.output_size[0]],
                                  activation=nn.Tanh(),
                                  output_activation=None)

        log_std = self._config.log_std if self._config.log_std != 'default' else 0.5
        self.log_std = torch.nn.Parameter(torch.ones(self.output_size, dtype=torch.float32) * log_std,
                                          requires_grad=True)

        self._critic = mlp_creator(sizes=[self.input_size[0], 64, 64, 1],
                                   activation=nn.Tanh(),
                                   output_activation=None)

        self.initialize_architecture()

    def initialize_architecture(self):
        torch.manual_seed(self._config.random_seed)
        torch.set_num_threads(1)
        for layer in self._actor[:-1]:
            initialize_weights(layer, initialisation_type=self._config.initialisation_type, scale=2**0.5)
        initialize_weights(self._actor[-1], initialisation_type=self._config.initialisation_type, scale=0.01)
        for layer in self._critic[:-1]:
            initialize_weights(layer, initialisation_type=self._config.initialisation_type, scale=2**0.5)
        initialize_weights(self._critic[-1], initialisation_type=self._config.initialisation_type, scale=1.0)

    def get_actor_parameters(self) -> Iterator:
        return list(self._actor.parameters()) + [self.log_std]

    def get_critic_parameters(self) -> Iterator:
        return self._critic.parameters()

    def _policy_distribution(self, inputs: torch.Tensor, train: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.process_inputs(inputs=inputs, train=train)
        means = self._actor(inputs)
        return means, torch.exp(self.log_std)

    def sample(self, inputs: torch.Tensor):
        mean, std = self._policy_distribution(inputs, train=False)
        return (mean + torch.randn_like(mean) * std).detach()

    def get_action(self, inputs) -> Action:
        output = self.sample(inputs)
        # output = output.clamp(min=self.action_min, max=self.action_max)
        return Action(actor_name=get_filename_without_extension(__file__),
                      value=output)

    def policy_log_probabilities(self, inputs, actions, train: bool = True) -> torch.Tensor:
        actions = self.process_inputs(inputs=actions, train=train)  # preprocess list of Actions
        try:
            mean, std = self._policy_distribution(inputs, train)
            log_probabilities = -(0.5 * ((actions - mean) / std).pow(2).sum(-1) +
                                  0.5 * np.log(2.0 * np.pi) * actions.shape[-1]
                                  + self.log_std.sum(-1))
            return log_probabilities
        except Exception as e:
            raise ValueError(f"Numerical error: {e}")

    def critic(self, inputs, train: bool = False) -> torch.Tensor:
        inputs = self.process_inputs(inputs=inputs, train=train)
        return self._critic(inputs)

    def get_policy_entropy(self, inputs: torch.Tensor, train: bool = True) -> torch.Tensor:
        mean, std = self._policy_distribution(inputs=inputs, train=train)
        return Normal(mean, std).entropy().sum(dim=1)
