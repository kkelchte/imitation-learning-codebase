from typing import List, Dict, Union
from enum import IntEnum

import numpy as np
from dataclasses import dataclass
import torch

from src.sim.common.data_types import Experience

"""Data types required for defining dataset frames and torch dataset samples.

Mainly used by data_saver for storing states as frames
 and data_loader for loading dataset frames as pytorch dataset samples.
Simulation related datatypes are specified in src/sim/common/data_types.py.
"""


def to_torch(value: Union[np.ndarray, int, float],
             dtype: torch.dtype = torch.float32):
    return torch.as_tensor(value, dtype=dtype)


@dataclass
class Dataset:  # Preparation for training DNN's in torch => only accept torch tensors
    observations: List[torch.Tensor] = None
    actions: List[torch.Tensor] = None
    rewards: List[torch.Tensor] = None
    done: List[torch.Tensor] = None  # 0, except on last episode step 1
    max_size: int = -1

    def __post_init__(self):
        if self.observations is None:
            self.observations = []
        if self.actions is None:
            self.actions = []
        if self.rewards is None:
            self.rewards = []
        if self.done is None:
            self.done = []

    def __len__(self):
        return len(self.observations)

    def pop(self):
        self.observations.pop(0)
        self.actions.pop(0)
        self.rewards.pop(0)
        self.done.pop(0)

    def append(self, experience: Experience):
        self.observations.append(to_torch(experience.observation))
        self.actions.append(to_torch(experience.action))
        self.rewards.append(to_torch(experience.reward))
        self.done.append(to_torch(experience.done))
        if len(self.observations) > self.max_size != -1:
            self.pop()

