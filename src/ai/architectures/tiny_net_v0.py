from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as functional

from src.ai.architectures.base_net import BaseNet
from src.ai.architectures.components import FourLayerReLuEncoder, ThreeLayerControlDecoder


class Net(BaseNet):

    def __init__(self, output_sizes: list = None, dropout: float = 0.0):
        super().__init__(dropout=dropout)
        self.output_sizes = output_sizes if output_sizes is not None else [[1]]

        self.encoder = FourLayerReLuEncoder()

        if self.dropout_rate:
            self.dropout = nn.Dropout(p=self._config.dropout_rate)

        self.decoder = ThreeLayerControlDecoder(output_size=self.output_sizes[0][-1])

        self.input_sizes = [(3, 128, 128)]

    def forward(self, inputs: List[torch.Tensor], train: bool = False) -> List[torch.Tensor]:
        """
        Outputs steering action only
        """
        if train:  # adjust gradient saving
            self.train()
        else:
            self.eval()
        x, = inputs
        x = self.encoder.forward(x)
        if self.dropout_rate != 0:
            x = self.dropout(x)
        x = self.decoder.forward(x)
        return [x]  # List because auxiliary task outputs should be added in more complex architectures
