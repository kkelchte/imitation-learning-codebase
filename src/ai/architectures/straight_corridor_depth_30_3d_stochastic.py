#!/bin/python3.8
from typing import Iterator

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from src.ai.architectures.bc_actor_critic_stochastic_discrete import Net as BaseNet
from src.ai.base_net import ArchitectureConfig
from src.ai.utils import mlp_creator, DiscreteActionMapper, initialize_weights
from src.core.data_types import Action
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension

"""
30 digits from laser scan.
3 actions: turn left, go straight, turn right.
"""


class Net(BaseNet):

    def __init__(self, config: ArchitectureConfig, quiet: bool = False):
        super().__init__(config=config, quiet=True)
        self.input_size = (30,)
        self.output_size = (3,)
        self._actor = mlp_creator(sizes=[self.input_size[0], 64, 64, self.output_size[0]],
                                  activation=nn.Tanh(),
                                  output_activation=None)

        self._critic = mlp_creator(sizes=[self.input_size[0], 64, 64, 1],
                                   activation=nn.Tanh(),
                                   output_activation=None)
        self.initialize_architecture()
        self.discrete_action_mapper = DiscreteActionMapper([
            torch.as_tensor([0.2, 0.0, 0.0, 0.0, 0.0, -0.2]),
            torch.as_tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
            torch.as_tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.2]),
        ])
        if not quiet:
            self._logger = get_logger(name=get_filename_without_extension(__file__),
                                      output_path=config.output_path,
                                      quiet=False)

            cprint(f'Started.', self._logger)

    def get_action(self, inputs, train: bool = False) -> Action:
        output = self._policy_distribution(inputs, train).sample().item()
        return Action(actor_name=get_filename_without_extension(__file__),
                      value=self.discrete_action_mapper.index_to_tensor(output))

    def policy_log_probabilities(self, inputs, actions, train: bool = True) -> torch.Tensor:
        actions = [self.discrete_action_mapper.tensor_to_index(a) for a in actions]
        actions = self.process_inputs(inputs=actions)
        return self._policy_distribution(inputs, train=train).log_prob(actions)
