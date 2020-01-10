from enum import IntEnum

from dataclasses import dataclass


@dataclass
class Action:
    pass


@dataclass
class OutcomeType(IntEnum):
    NotDone = 0
    Success = 1
    Failure = 2


@dataclass
class State:
    outcome: OutcomeType = None


class EnvironmentType(IntEnum):
    Gym = 0
    Gazebo = 1
    Real = 2


class ActorType(IntEnum):
    Model = 0
    Expert = 1
