from typing import List, Dict
from enum import IntEnum

import numpy as np
from dataclasses import dataclass
import torch


"""Data types required for defining dataset frames and torch dataset samples.

Mainly used by data_saver for storing states as frames
 and data_loader for loading dataset frames as pytorch dataset samples.
Simulation related datatypes are specified in src/sim/common/data_types.py.
"""


@dataclass
class OutcomeType(IntEnum):
    Failure = 0
    Success = 1


@dataclass
class Frame:
    origin: str = None
    time_stamp_ms: int = None
    data: np.array = None


@dataclass
class Episode:
    frames: List[Frame] = None
    outcome: OutcomeType = None

    def __len__(self):
        return len(self.frames)


@dataclass
class Run:
    outputs: Dict[str, torch.Tensor] = None
    inputs: Dict[str, torch.Tensor] = None
    reward: torch.Tensor = None

    def __post_init__(self):
        if self.outputs is None:
            self.outputs = {}
        if self.inputs is None:
            self.inputs = {}
        if self.reward is None:
            self.reward = torch.Tensor()

    def __len__(self):
        return max([len(o) for o in self.outputs.values()] +
                   [len(i) for i in self.inputs.values()] + [len(self.reward)])

    def get_input(self) -> list:
        return list(self.inputs.values())

    def get_output(self) -> list:
        return list(self.outputs.values())


@dataclass
class Dataset:
    data: List[Run] = None

    def __post_init__(self):
        if self.data is None:
            self.data = []

    def __len__(self):
        return len(self.data)
