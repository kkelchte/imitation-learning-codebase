#!/bin/python3.8
from typing import Iterator

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from src.ai.base_net import BaseNet, ArchitectureConfig
from src.ai.utils import mlp_creator, DiscreteActionMapper, initialize_weights
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

        self.input_size = (30,)
        self.output_size = (3,)
        self.discrete = True
        self._actor = mlp_creator(sizes=[self.input_size[0], 64, 64, self.output_size[0]],
                                  activation=nn.Tanh,
                                  output_activation=None)

        self._critic = mlp_creator(sizes=[self.input_size[0], 64, 64, 1],
                                   activation=nn.Tanh,
                                   output_activation=None)
        self.initialize_architecture()
        self.discrete_action_mapper = DiscreteActionMapper([
            torch.as_tensor([0.2, 0.0, 0.0, 0.0, 0.0, -0.2]),
            torch.as_tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
            torch.as_tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.2]),
        ])

    def initialize_architecture(self):
        torch.manual_seed(self._config.random_seed)
        torch.set_num_threads(1)
        for layer in self._actor[:-1]:
            initialize_weights(layer, initialisation_type=self._config.initialisation_type, scale=2**0.5)
        initialize_weights(self._actor[-1], initialisation_type=self._config.initialisation_type, scale=1.)
        for layer in self._critic[:-1]:
            initialize_weights(layer, initialisation_type=self._config.initialisation_type, scale=2**0.5)
        initialize_weights(self._critic[-1], initialisation_type=self._config.initialisation_type, scale=1.)

    def get_actor_parameters(self) -> Iterator:
        return self._actor.parameters()

    def get_critic_parameters(self) -> Iterator:
        return self._critic.parameters()

    def _policy_distribution(self, inputs: torch.Tensor, train: bool = True) -> Categorical:
        inputs = super().forward(inputs=inputs, train=train)
        logits = self._actor(inputs)
        return Categorical(logits=logits)

    def get_policy_entropy(self, inputs: torch.Tensor, train: bool = True) -> torch.Tensor:
        distribution = self._policy_distribution(inputs=inputs, train=train)
        return -(distribution.probs * torch.log(distribution.probs)).sum(dim=1)

    def get_action(self, inputs, train: bool = False) -> Action:
        output = self._policy_distribution(inputs, train).sample().item()
        return Action(actor_name=get_filename_without_extension(__file__),
                      value=self.discrete_action_mapper.index_to_tensor(output))

    def policy_log_probabilities(self, inputs, actions, train: bool = True) -> torch.Tensor:
        actions = [self.discrete_action_mapper.tensor_to_index(a) for a in actions]
        actions = super().forward(inputs=actions, train=train)
        return self._policy_distribution(inputs, train=train).log_prob(actions)

    def critic(self, inputs, train: bool = False) -> torch.Tensor:
        inputs = super().forward(inputs=inputs, train=train)
        return self._critic(inputs)
