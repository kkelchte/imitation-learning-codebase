from typing import Dict
from enum import IntEnum

import numpy as np
from dataclasses import dataclass


class TerminalType(IntEnum):
    NotDone = 0
    Success = 1
    Failure = 2


class EnvironmentType(IntEnum):
    Ros = 0
    Gym = 1
    Real = 2


class ActorType(IntEnum):
    Model = 0
    Expert = 1


class SensorType(IntEnum):
    RGB = 0
    Depth = 1


@dataclass
class Action:
    actor_type: ActorType = None
    value: np.ndarray = None


@dataclass
class State:
    terminal: TerminalType = None
    sensor_data: Dict[SensorType, np.ndarray] = None
    time_stamp_us: int = 0
