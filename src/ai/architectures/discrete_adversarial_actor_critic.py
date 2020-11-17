#!/bin/python3.8
from typing import Iterator, Union, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from torch.distributions.categorical import Categorical

from src.ai.architectures.bc_actor_critic_stochastic_discrete import Net as BaseNet
from src.ai.base_net import ArchitectureConfig
from src.ai.utils import mlp_creator, get_slow_hunt, DiscreteActionMapper
from src.core.data_types import Action
from src.core.logger import get_logger, cprint, MessageType
from src.core.utils import get_filename_without_extension

"""
DummyTrackingEnv-v0	action space: discrete 5 observation space: Discrete(5)
observation space is shared by standard (tracking) and adversarial (fleeing) agent.
First two action values correspond to the blue or default agent who tracks the red square.
The second two action values move the red or adversary agent who flees from the blue square.
"""

EPSILON = 1e-6


class Net(BaseNet):

    def __init__(self, config: ArchitectureConfig, quiet: bool = False):
        super().__init__(config=config, quiet=True)
        self.input_size = (4,)
        self.output_size = (5,)
        self.discrete = True

        self._actor = mlp_creator(sizes=[self.input_size[0], 10, self.output_size[0]],
                                  activation=nn.Tanh(),
                                  output_activation=None)

        self._critic = mlp_creator(sizes=[self.input_size[0], 10, 1],
                                   activation=nn.Tanh(),
                                   output_activation=None)

        log_std = self._config.log_std if self._config.log_std != 'default' else -0.5
        self.log_std = torch.nn.Parameter(torch.ones(self.output_size, dtype=torch.float32) * log_std,
                                          requires_grad=True)

        self._adversarial_actor = mlp_creator(sizes=[self.input_size[0], 10, self.output_size[0]],
                                              activation=nn.Tanh(),
                                              output_activation=None)

        self._adversarial_critic = mlp_creator(sizes=[self.input_size[0], 10, 1],
                                               activation=nn.Tanh(),
                                               output_activation=None)

        self.adversarial_log_std = torch.nn.Parameter(torch.ones(self.output_size,
                                                                 dtype=torch.float32) * log_std, requires_grad=True)

        self.discrete_action_mapper = DiscreteActionMapper([
            torch.as_tensor([0.0, 0.0]),
            torch.as_tensor([-1.0, 0.0]),
            torch.as_tensor([1.0, 0.0]),
            torch.as_tensor([0.0, -1.0]),
            torch.as_tensor([0.0, 1.0]),
        ])

        if not quiet:
            self._logger = get_logger(name=get_filename_without_extension(__file__),
                                      output_path=config.output_path,
                                      quiet=False)

            cprint(f'Started.', self._logger)

            self.initialize_architecture()

    def get_action(self, inputs, train: bool = False, agent_id: int = -1) -> Action:
        inputs = self.process_inputs(inputs)
        if agent_id == 0:
            output = self.sample(inputs, train=train, adversarial=False)
            action = self.discrete_action_mapper.index_to_tensor(output)
            actions = np.stack([*action.data.cpu().numpy().squeeze(), 0, 0])
        elif agent_id == 1:
            output = self.sample(inputs, train=train, adversarial=True)
            action = self.discrete_action_mapper.index_to_tensor(output)
            actions = np.stack([*get_slow_hunt(inputs.squeeze()),
                                *action.data.cpu().numpy().squeeze()], axis=-1)
        else:
            output = self.sample(inputs, train=train, adversarial=False)
            tracking_action = self.discrete_action_mapper.index_to_tensor(output)
            adversarial_output = self.sample(inputs, train=train, adversarial=True)
            adversarial_action = self.discrete_action_mapper.index_to_tensor(adversarial_output)
            actions = np.stack([*tracking_action.data.cpu().numpy().squeeze(),
                                *adversarial_action.data.cpu().numpy().squeeze()], axis=-1)
        return Action(actor_name=get_filename_without_extension(__file__),  # assume output [1, 2] so no batch!
                      value=actions)

    def _policy_distribution(self, inputs: torch.Tensor, train: bool = True, adversarial: bool = False) -> Categorical:
        self.set_mode(train)
        inputs = self.process_inputs(inputs=inputs)
        logits = self._actor(inputs) if not adversarial else self._adversarial_actor(inputs)
        return Categorical(logits=logits)

    def sample(self, inputs: torch.Tensor, train: bool = False, adversarial: bool = False) -> int:
        distribution = self._policy_distribution(inputs, train=train, adversarial=adversarial)
        return distribution.sample()

    def get_adversarial_policy_entropy(self, inputs: torch.Tensor, train: bool = True) -> torch.Tensor:
        distribution = self._policy_distribution(inputs=inputs, train=train, adversarial=True)
        return distribution.entropy()

    def policy_log_probabilities(self, inputs, actions, train: bool = True, adversarial: bool = False) -> torch.Tensor:
        actions = torch.as_tensor([self.discrete_action_mapper.tensor_to_index(a[:2]) if not adversarial
                                   else self.discrete_action_mapper.tensor_to_index(a[2:]) for a in actions])
        actions = self.process_inputs(inputs=actions)
        try:
            return self._policy_distribution(inputs, train=train, adversarial=adversarial).log_prob(actions)
        except Exception as e:
            raise ValueError(f"Numerical error: {e}")

    def adversarial_policy_log_probabilities(self, inputs, actions, train: bool = True) -> torch.Tensor:
        return self.policy_log_probabilities(inputs, actions, train, adversarial=True)

    def adversarial_critic(self, inputs, train: bool = False) -> torch.Tensor:
        self._adversarial_critic.train(train)
        inputs = self.process_inputs(inputs=inputs)
        return self._adversarial_critic(inputs)

    def get_adversarial_actor_parameters(self) -> Iterator:
        return self._adversarial_actor.parameters()

    def get_adversarial_critic_parameters(self) -> Iterator:
        return self._adversarial_critic.parameters()
