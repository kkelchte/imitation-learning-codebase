from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as functional

from src.ai.architectures.base_net import BaseNet


class Net(BaseNet):

    def __init__(self, dropout: float = 0.0):
        super().__init__(dropout=dropout)

        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        if self.dropout_rate:
            self.dropout = nn.Dropout(p=self._config.dropout_rate)

        self.fc1 = nn.Linear(2 * 2 * 256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1, bias=False)

        self.input_sizes = [(3, 128, 128)]

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Outputs steering action only
        """
        x, = inputs
        x = functional.relu(self.conv1(x))
        x = functional.relu(self.conv2(x))
        x = functional.relu(self.conv3(x))
        x = functional.relu(self.conv4(x))

        x = x.view(-1, 2 * 2 * 256)
        if self.dropout_rate != 0:
            x = self.dropout(x)
        x = functional.relu(self.fc1(x))
        if self.dropout_rate != 0:
            x = self.dropout(x)
        x = functional.relu(self.fc2(x))
        if self.dropout_rate != 0:
            x = self.dropout(x)
        x = self.fc3(x)
        return x
