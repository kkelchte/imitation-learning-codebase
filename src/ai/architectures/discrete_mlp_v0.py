from typing import List

import torch

from src.ai.architectures.base_net import BaseNet
from src.ai.architectures.components import ThreeLayerPerceptron


class Net(BaseNet):

    def __init__(self, output_sizes: List[List] = None, dropout: float = 0.0):
        super().__init__(dropout=dropout)
        self.input_sizes = [[4]]
        self.output_sizes = output_sizes if output_sizes is not None else [[2]]
        self.mlp = ThreeLayerPerceptron(self.input_sizes[0][0], self.output_sizes[0][0])

    def forward(self, inputs: List[torch.Tensor], train: bool = False) -> List[torch.Tensor]:
        """
        Outputs steering action only
        """
        if train:  # adjust gradient saving
            self.train()
        else:
            self.eval()
        x, = inputs
        x = self.mlp.forward(x)
        return [x]
