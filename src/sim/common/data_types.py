from enum import IntEnum

from dataclasses import dataclass


@dataclass
class Action:
    pass


@dataclass
class State:
    pass


class EnvironmentType(IntEnum):
    Gym = 0
    Gazebo = 1
    Real = 2
