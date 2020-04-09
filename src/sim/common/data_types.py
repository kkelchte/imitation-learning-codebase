from typing import Dict
from enum import IntEnum

import numpy as np
from dataclasses import dataclass


class ProcessState(IntEnum):
    Running = 0
    Terminated = 1
    Unknown = 2
    Initializing = 3


class TerminalType(IntEnum):
    Unknown = -1
    NotDone = 0
    Done = 1
    Success = 2
    Failure = 3


class EnvironmentType(IntEnum):
    Ros = 0
    Gym = 1
    Real = 2


class ActorType(IntEnum):
    Unknown = -1
    Model = 0
    Expert = 1
    User = 2


@dataclass
class Action:
    actor_type: ActorType = None
    actor_name: str = None
    value: np.ndarray = None

    def __len__(self):
        return len(self.value) if self.value is not None else 0


@dataclass
class State:
    terminal: TerminalType = None
    actor_data: Dict[str, Action] = None
    sensor_data: Dict[str, np.ndarray] = None
    time_stamp_ms: int = 0
