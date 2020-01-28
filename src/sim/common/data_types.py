from typing import Dict
from enum import IntEnum

import numpy as np
from dataclasses import dataclass


class TerminalType(IntEnum):
    Unknown = -1
    NotDone = 0
    Success = 1
    Failure = 2


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


@dataclass
class State:
    terminal: TerminalType = None
    actor_data: Dict[str, Action] = None
    sensor_data: Dict[str, np.ndarray] = None
    time_stamp_us: int = 0
