import torch
from torch import nn
import torch.nn.functional as functional


class ArchiterturalComponentBase:

    def __init__(self):
        self.input = (3, 100, 100)
        self.output = (1, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class FourLayerReLuEncoder(ArchiterturalComponentBase):

    def __init__(self):
        super().__init__()
        self.input = (3, 128, 128)
        self.output = (1, 1, 2 * 2 * 256)

        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = functional.relu(self.conv1(x))
        x = functional.relu(self.conv2(x))
        x = functional.relu(self.conv3(x))
        x = functional.relu(self.conv4(x))
        return x.view(-1, 2 * 2 * 256)


class ThreeLayerControlDecoder(ArchiterturalComponentBase):

    def __init__(self, output_size: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(2 * 2 * 256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        return self.fc3(x)


class DepthDecoder(ArchiterturalComponentBase):

    def __init__(self):
        super().__init__()
