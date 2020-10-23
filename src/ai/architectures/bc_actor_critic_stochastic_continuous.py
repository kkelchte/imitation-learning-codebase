#!/bin/python3.8
from typing import Iterator, Tuple, Union

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
Base Class used by continuous stochastic actor-critic networks.
"""


class Net(BaseNet):

    def __init__(self, config: ArchitectureConfig, quiet: bool = False):
        super().__init__(config=config, quiet=True)
        self.discrete = False
        if not quiet:
            NotImplementedError('This class is supposed to be used as baseclass (quiet=False)')

    def initialize_architecture(self):
        cprint(f'using scaled initialization', logger=self._logger)
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
        self.set_mode(train)
        inputs = self.process_inputs(inputs=inputs)
        means = self._actor(inputs)
        return means, torch.exp(self.log_std)

    def sample(self, inputs: torch.Tensor, train: bool = False):
        mean, std = self._policy_distribution(inputs, train=train)
        return (mean + torch.randn_like(mean) * std).detach()

    def get_action(self, inputs, train: bool = False) -> Action:
        inputs = self.process_inputs(inputs)  # added line 15/10/2020
        output = self.sample(inputs, train=train)
        output = output.clamp(min=self.action_min, max=self.action_max)  # added line 23/10/2020
        return Action(actor_name=get_filename_without_extension(__file__),
                      value=output)

    def policy_log_probabilities(self, inputs, actions, train: bool = True) -> torch.Tensor:
        actions = self.process_inputs(inputs=actions)  # preprocess list of Actions
        try:
            mean, std = self._policy_distribution(inputs, train)
            log_probabilities = -(0.5 * ((actions - mean) / std).pow(2).sum(-1) +
                                  0.5 * np.log(2.0 * np.pi) * actions.shape[-1]
                                  + self.log_std.sum(-1))
            return log_probabilities
        except Exception as e:
            raise ValueError(f"Numerical error: {e}")

    def critic(self, inputs, train: bool = False) -> torch.Tensor:
        self._critic.train() if train else self._critic.eval()
        inputs = self.process_inputs(inputs=inputs)
        return self._critic(inputs)

    def get_policy_entropy(self, inputs: torch.Tensor, train: bool = True) -> torch.Tensor:
        mean, std = self._policy_distribution(inputs=inputs, train=train)
        return Normal(mean, std).entropy().sum(dim=1)

    # def set_device(self, device: Union[str, torch.device]):
    #     self._device = torch.device(
    #         "cuda" if device in ['gpu', 'cuda'] and torch.cuda.is_available() else "cpu"
    #     ) if isinstance(device, str) else device
    #     try:
    #         self.to(self._device)
    #         self._actor.to(self._device)
    #         self._critic.to(self._device)
    #     except AssertionError:
    #         cprint(f'failed to work on {self._device} so working on cpu', self._logger, msg_type=MessageType.warning)
    #         self._device = torch.device('cpu')
    #         self.to(self._device)
