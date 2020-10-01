import sys
from typing import Dict, Union, List, Iterator
from enum import IntEnum

import h5py
import numpy as np
import torch
from dataclasses import dataclass


class Distribution:
    mean: float = 0
    std: float = 0
    min: float = 0
    max: float = 0

    def __init__(self, data: Iterator):
        if isinstance(data, list):
            try:
                if isinstance(data[0], torch.Tensor):
                    data = torch.stack(data)
            except IndexError:
                self.mean = 0
                self.std = 0
                self.min = 0
                self.max = 0
                return
        if isinstance(data, torch.Tensor):
            if not isinstance(data, torch.FloatTensor):
                data = data.type(torch.float32)
            self.mean = data.mean().item()
            self.std = data.std().item()
            self.min = data.min().item()
            self.max = data.max().item()
        else:
            if not isinstance(data, np.ndarray):
                data = np.asarray(data)
            self.mean = np.mean(data).item()
            self.std = np.std(data).item()
            self.min = np.min(data)
            self.max = np.max(data)
        # assert not np.isnan(self.mean)


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


def to_torch(value: Union[np.ndarray, int, float, torch.Tensor],
             dtype: torch.dtype = None):
    if isinstance(value, torch.Tensor):
        return value if dtype is None else value.type(dtype)
    else:
        return torch.as_tensor(value, dtype=dtype if dtype is not None else torch.float32) \
            if value is not None else None


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

    def _get_size_of_field(self, data: List[torch.Tensor]) -> int:
        if data is None or len(data) == 0 or None in data:
            return 0
        num_elements = len(data) * np.prod(data[0].shape)
        try:
            byte_number = int(str(data[0].dtype)[-2:])
        except ValueError:
            byte_number = 32
        return int(byte_number * num_elements / 8)

    def get_memory_size(self) -> int:
        return sum(self._get_size_of_field(d) for d in [self.observations, self.actions, self.rewards, self.done])

    def append(self, experience: Experience):
        self.observations.append(to_torch(experience.observation))
        self.actions.append(to_torch(experience.action.value
                                     if isinstance(experience.action, Action) else experience.action))
        self.rewards.append(to_torch(experience.reward))
        self.done.append(to_torch(experience.done))
        self._check_length()

    def pop(self):
        """pop oldest experience"""
        self.observations.pop(0)
        self.actions.pop(0)
        self.rewards.pop(0)
        self.done.pop(0)

    def clip(self, length: int):
        """clip dataset sequence at a fixed length at the end of the sequence"""
        if self.__len__() > length:
            self.observations = self.observations[:length]
            self.actions = self.actions[:length]
            self.rewards = self.rewards[:length]
            self.done = self.done[:length]

    def _check_length(self):
        """remove oldest experience to fit in max_size of buffer"""
        while len(self) > self.max_size != -1:
            self.pop()

    def extend(self, experiences: Union[List[Experience], h5py.Group, "Dataset"]):
        """
        Extend the dataset with multiple experiences rather than appending one.
        Experiences can be a loaded H5py group, a list of experiences or a dataset.
        The dataset type is defined as a string so the typing is only interpreted after compilation.
        See:
        https://stackoverflow.com/questions/44798635/how-can-i-set-the-same-type-as-class-in-methods-parameter-following-pep484
        """
        if isinstance(experiences, h5py.Group):
            for tag, field in zip(['observations', 'actions', 'rewards', 'done'],
                                  [self.observations, self.actions, self.rewards, self.done]):
                if tag in experiences.keys():
                    field.extend([torch.as_tensor(v, dtype=torch.float32) for v in experiences[tag]])
                else:
                    field.extend([torch.zeros(0) for _ in experiences['observations']])
            self._check_length()
        elif isinstance(experiences, Dataset):
            for data, field in zip([experiences.observations, experiences.actions, experiences.rewards,
                                    experiences.done],
                                   [self.observations, self.actions, self.rewards, self.done]):
                if len(data) != 0:
                    field.extend(data)
                else:
                    field.extend([torch.zeros(0) for _ in range(len(experiences))])
            self._check_length()
        else:
            for exp in experiences:
                self.append(exp)
            self._check_length()

    def subsample(self, subsample: int):
        to_be_deleted_indices = []
        index = 0
        run_step = 0
        while index < len(self):
            if run_step % subsample != 0 and self.done[index].item() == 0:
                to_be_deleted_indices.append(index)
            if self.done[index].item() != 0:
                run_step = 0
            else:
                run_step += 1
            index += 1

        for index in reversed(to_be_deleted_indices):
            del self.observations[index]
            del self.actions[index]
            del self.rewards[index]
            del self.done[index]

    def shuffle(self):
        """shuffles data"""
        indices = list(range(len(self)))
        np.random.shuffle(indices)
        self.observations = [self.observations[i] for i in indices]
        self.actions = [self.actions[i] for i in indices]
        self.rewards = [self.rewards[i] for i in indices]
        self.done = [self.done[i] for i in indices]
