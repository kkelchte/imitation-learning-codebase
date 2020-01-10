from typing import List, Dict
from enum import IntEnum

import numpy as np
from dataclasses import dataclass
import torch

"""Data types required for defining dataset frames and torch dataset samples.

Mainly used by data_saver for storing states as dataset frames
 and data_loader for loading dataset frames as pytorch dataset samples.
Simulation related datatypes are specified in src/sim/common/data_types.py.
"""


@dataclass
class OutcomeType(IntEnum):
    Failure = 0
    Success = 1


@dataclass
class Frame:
    sensor: str = None
    time_stamp: int = None
    data: np.array = None


@dataclass
class Episode:
    frames: List[Frame] = None
    outcome: OutcomeType = None

    def __len__(self):
        return len(self.frames)


@dataclass
class Dataset:
    episodes: List[Episode] = None


@dataclass
class Sample:
    """Contains all relevant sensor readings related to one time stamp."""
    label: torch.Tensor = None
    input: torch.Tensor = None
    reward: torch.Tensor = None
    auxiliary_input: Dict[torch.Tensor] = None
    auxiliary_output: Dict[torch.Tensor] = None


@dataclass
class TorchDataset:
    data: List[Sample]
