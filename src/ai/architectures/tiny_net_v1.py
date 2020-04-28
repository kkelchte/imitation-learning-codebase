from typing import List

import torch
import torch.nn as nn

from src.ai.architectures.base_net import BaseNet, GetItem
from src.ai.architectures.components import FourLayerReLuEncoder, ThreeLayerControlDecoder


class Net(BaseNet):

    def __init__(self, input_sizes: GetItem, output_sizes: GetItem, dropout: float = 0.0):
        super().__init__(input_sizes=input_sizes, output_sizes=output_sizes)
        self.dropout = nn.Dropout(p=dropout) if dropout != 0.0 else None

        #  To check: if elements gradients are set to zero correctly + to eval/train mode
        self.encoder = FourLayerReLuEncoder()
        self.decoder = ThreeLayerControlDecoder(output_size=self.output_sizes[0][0])

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
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.decoder.forward(x)
        return [x]
