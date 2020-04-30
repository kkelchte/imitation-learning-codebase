from typing import Dict, Union, List
from enum import IntEnum

import h5py
import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class Distribution:
    mean: float = 0
    std: float = 0


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
    done: Union[int, np.ndarray, torch.Tensor, TerminationType] = None
    time_stamp: int = 999
    info: Dict = None


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

    def append(self, experience: Experience):
        self.observations.append(to_torch(experience.observation))
        self.actions.append(to_torch(experience.action))
        self.rewards.append(to_torch(experience.reward))
        self.done.append(to_torch(experience.done))
        self._check_length()

    def pop(self):
        self.observations.pop(0)
        self.actions.pop(0)
        self.rewards.pop(0)
        self.done.pop(0)

    def _check_length(self):
        while len(self) > self.max_size != -1:
            self.pop()

    def extend(self, experiences: Union[List[Experience], h5py.Group]):
        if isinstance(experiences, h5py.Group):
            self.observations.extend([torch.as_tensor(v, dtype=torch.float32) for v in experiences['observations']])
            self.actions.extend([torch.as_tensor(v, dtype=torch.float32) for v in experiences['actions']])
            self.rewards.extend([torch.as_tensor(v, dtype=torch.float32) for v in experiences['rewards']])
            self.done.extend([torch.as_tensor(v, dtype=torch.float32) for v in experiences['done']])
            self._check_length()
        else:
            for exp in experiences:
                self.append(exp)
            self._check_length()
