
from dataclasses import dataclass

from src.sim.common.environment import EnvironmentConfig, Environment


@dataclass
class GymEnvironmentConfig:
    pass


class GymEnvironment:

    def __init__(self):
        super().__init__()

