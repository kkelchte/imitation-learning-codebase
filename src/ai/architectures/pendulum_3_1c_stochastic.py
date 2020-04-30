#!/bin/python3.7

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from src.ai.base_net import BaseNet, ArchitectureConfig
from src.ai.utils import mlp_creator
from src.core.data_types import Action
from src.core.utils import get_filename_without_extension

"""
Pendulum-v0	action space: continuous 1[-2.0 : 2.0]	observation space: Box(3,)
"""


class Net(BaseNet):

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config=config)
        self.input_size = (4,)
        self.output_size = (2,)
        self.mlp = mlp_creator(sizes=[self.input_size[0], 25, 25, self.output_size[0]],
                               activation=nn.ReLU,
                               output_activation=None)

    def _distribution(self, inputs: torch.Tensor) -> Categorical:
        logits = self.mlp(inputs)
        return Categorical(logits=logits)

    def forward(self, inputs, train: bool = False) -> torch.Tensor:
        """
        Outputs steering action only
        """
        inputs = super().forward(inputs=inputs, train=train)
        return self._distribution(inputs).sample()

    def get_action(self, inputs, train: bool = False) -> Action:
        output = self.forward(inputs, train)
        return Action(actor_name=get_filename_without_extension(__file__),
                      value=output.item())
