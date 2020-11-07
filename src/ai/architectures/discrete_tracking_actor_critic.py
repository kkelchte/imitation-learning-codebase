#!/bin/python3.8
from typing import Iterator, Union

import torch
import torch.nn as nn
import numpy as np

from src.ai.architectures.bc_actor_critic_stochastic_discrete import Net as BaseNet
from src.ai.base_net import ArchitectureConfig
from src.ai.utils import mlp_creator, DiscreteActionMapper
from src.core.data_types import Action
from src.core.logger import get_logger, cprint, MessageType
from src.core.utils import get_filename_without_extension

"""
discrete actor critic with 4d actions (X,Y) for 2 agents
"""


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

        self.initialize_architecture()

        self.discrete_action_mapper = DiscreteActionMapper([
            torch.as_tensor([0.0, 0.0, 0.0, 0.0]),
            torch.as_tensor([-1.0, 0.0, 0.0, 0.0]),
            torch.as_tensor([1.0, 0.0, 0.0, 0.0]),
            torch.as_tensor([0.0, -1.0, 0.0, 0.0]),
            torch.as_tensor([0.0, 1.0, 0.0, 0.0]),
        ])
        if not quiet:
            self._logger = get_logger(name=get_filename_without_extension(__file__),
                                      output_path=config.output_path,
                                      quiet=False)

            cprint(f'Started.', self._logger)

    def get_action(self, inputs, train: bool = False) -> Action:
        output = self._policy_distribution(inputs, train).sample().item()
        return Action(actor_name=get_filename_without_extension(__file__),
                      value=np.array(self.discrete_action_mapper.index_to_tensor(output)))

    def policy_log_probabilities(self, inputs, actions, train: bool = True) -> torch.Tensor:
        actions = torch.as_tensor([self.discrete_action_mapper.tensor_to_index(a) for a in actions])
        actions = self.process_inputs(inputs=actions)
        return self._policy_distribution(inputs, train=train).log_prob(actions)
