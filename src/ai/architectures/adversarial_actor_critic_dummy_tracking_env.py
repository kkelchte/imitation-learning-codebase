#!/bin/python3.8
from typing import Iterator

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from torch.distributions.categorical import Categorical

from src.ai.base_net import BaseNet, ArchitectureConfig
from src.ai.utils import mlp_creator
from src.core.data_types import Action
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension

"""
DummyTrackingEnv-v0	action space: continuous 4	observation space: Box(4,)
observation space is shared by standard (tracking) and adversarial (fleeing) agent.
First two action values correspond to the blue or default agent who tracks the red square.
The second two action values move the red or adversary agent who flees from the blue square.
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
        self.output_size = (4,)
        self.action_min = -1
        self.action_max = 1

        log_std = self._config.log_std if self._config.log_std != 'default' else 1.0
        self.log_std = torch.nn.Parameter(torch.as_tensor([log_std] * 2, dtype=torch.float), requires_grad=True)

        self.discrete = False
        self._actor = mlp_creator(sizes=[self.input_size[0], 10, 2],
                                  activation=nn.Tanh(),
                                  output_activation=None)

        self._critic = mlp_creator(sizes=[self.input_size[0], 10, 1],
                                   activation=nn.Tanh(),
                                   output_activation=None)

        self._adversarial_actor = mlp_creator(sizes=[self.input_size[0], 10, 2],
                                              activation=nn.Tanh(),
                                              output_activation=None)

        self._adversarial_critic = mlp_creator(sizes=[self.input_size[0], 10, 1],
                                               activation=nn.Tanh(),
                                               output_activation=None)

        self.initialize_architecture()

    def get_action(self, inputs, train: bool = False) -> Action:
        output = self._policy_distribution(inputs, train).sample()
        output = output.clamp(min=self.action_min, max=self.action_max)
        adversarial_output = self._adversarial_policy_distribution(inputs, train).sample()
        adversarial_output = adversarial_output.clamp(min=self.action_min, max=self.action_max)
        actions = np.stack([*output.data.cpu().numpy().squeeze(),
                            *adversarial_output.data.cpu().numpy().squeeze()], axis=-1)
        return Action(actor_name=get_filename_without_extension(__file__),  # assume output [1, 2] so no batch!
                      value=actions)

    def _policy_distribution(self, inputs: torch.Tensor, train: bool = True, adversarial: bool = False) -> Normal:
        inputs = self.process_inputs(inputs=inputs, train=train)
        logits = self._actor(inputs) if not adversarial else self._adversarial_actor(inputs)
        return Normal(logits, torch.exp(self.log_std))

    def get_policy_entropy(self, inputs: torch.Tensor, train: bool = True) -> torch.Tensor:
        distribution = self._policy_distribution(inputs=inputs, train=train)
        return distribution.entropy().sum(dim=1)

    def get_adversarial_policy_entropy(self, inputs: torch.Tensor, train: bool = True) -> torch.Tensor:
        distribution = self._policy_distribution(inputs=inputs, train=train, adversarial=True)
        return distribution.entropy().sum(dim=1)

    def policy_log_probabilities(self, inputs, actions, train: bool = True) -> torch.Tensor:
        actions = self.process_inputs(inputs=actions, train=train)
        distribution = self._policy_distribution(inputs, train=train)
        return distribution.log_prob(actions[:, :2]).sum(-1)

    def adversarial_policy_log_probabilities(self, inputs, actions, train: bool = True) -> torch.Tensor:
        actions = self.process_inputs(inputs=actions, train=train)
        distribution = self._policy_distribution(inputs, train=train, adversarial=True)
        return distribution.log_prob(actions[:, :2]).sum(-1)

    def critic(self, inputs, train: bool = False) -> torch.Tensor:
        inputs = self.process_inputs(inputs=inputs, train=train)
        return self._critic(inputs)

    def adversarial_critic(self, inputs, train: bool = False) -> torch.Tensor:
        inputs = self.process_inputs(inputs=inputs, train=train)
        return self._adversarial_critic(inputs)

    def get_actor_parameters(self) -> Iterator:
        return self._actor.parameters()

    def get_critic_parameters(self) -> Iterator:
        return self._critic.parameters()

    def get_adversarial_actor_parameters(self) -> Iterator:
        return self._adversarial_actor.parameters()

    def get_adversarial_critic_parameters(self) -> Iterator:
        return self._adversarial_critic.parameters()

