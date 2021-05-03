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
Base Class used by discrete stochastic actor-critic networks.
"""


class Net(BaseNet):

    def __init__(self, config: ArchitectureConfig, quiet: bool = False):
        super().__init__(config=config, quiet=True)
        self.discrete = True
        if not quiet:
            self._logger = get_logger(name=get_filename_without_extension(__file__),
                                      output_path=config.output_path,
                                      quiet=False)

            cprint(f'Started.', self._logger)
            self.initialize_architecture()

    def get_actor_parameters(self) -> Iterator:
        return self._actor.parameters()

    def get_critic_parameters(self) -> Iterator:
        return self._critic.parameters()

    def _policy_distribution(self, inputs: torch.Tensor, train: bool = True) -> Categorical:
        self.set_mode(train)
        inputs = self.process_inputs(inputs=inputs)
        logits = nn.functional.softmax(self._actor(inputs))
        return Categorical(logits=logits)

    def get_policy_entropy(self, inputs: torch.Tensor, train: bool = True) -> torch.Tensor:
        distribution = self._policy_distribution(inputs=inputs, train=train)
        return -(distribution.probs * torch.log(distribution.probs)).sum(dim=1)

    def get_action(self, inputs, train: bool = False) -> Action:
        output = self._policy_distribution(inputs, train).sample()
        return Action(actor_name=get_filename_without_extension(__file__),
                      value=output.item())

    def policy_log_probabilities(self, inputs, actions, train: bool = True) -> torch.Tensor:
        actions = self.process_inputs(inputs=actions)
        return self._policy_distribution(inputs, train=train).log_prob(actions)

    def critic(self, inputs, train: bool = False) -> torch.Tensor:
        self._critic.train() if train else self._critic.eval()
        inputs = self.process_inputs(inputs=inputs)
        return self._critic(inputs)
