from typing import Dict, Union
from enum import IntEnum

import numpy as np
import torch
from dataclasses import dataclass


class ProcessState(IntEnum):
    Running = 0
    Terminated = 1
    Unknown = 2
    Initializing = 3


class TerminationType(IntEnum):
    Unknown = -1
    NotDone = 0
    Done = 1
    Success = 2
    Failure = 3


class EnvironmentType(IntEnum):
    Ros = 0
    Gym = 1
    Real = 2


@dataclass
class Action:
    actor_name: str = ''
    value: Union[int, float, np.ndarray, torch.Tensor] = None

    def __len__(self):
        return len(self.value) if self.value is not None else 0


@dataclass
class Experience:
    observation: Union[np.ndarray, torch.Tensor] = None
    action: Union[int, float, np.ndarray, torch.Tensor, Action] = None
    reward: Union[int, float, np.ndarray, torch.Tensor] = None
    done: TerminationType = None
    time_stamp: int = 999
    info: Dict = None
