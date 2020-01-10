from enum import IntEnum

from dataclasses import dataclass


@dataclass
class Action:
    pass


@dataclass
class TerminalType(IntEnum):
    NotDone = 0
    Success = 1
    Failure = 2


@dataclass
class State:
    terminal: TerminalType = None
    sensor_data: dict = None
    time_stamp: int = 0


class EnvironmentType(IntEnum):
    Gym = 0
    Gazebo = 1
    Real = 2


class ActorType(IntEnum):
    Model = 0
    Expert = 1
