
from dataclasses import dataclass

from src.sim.common.environment import EnvironmentConfig
from src.sim.environment_factory import EnvironmentFactory


@dataclass
class EnvironmentRunnerConfig:
    environment_config: EnvironmentConfig
    environment_type: str
    number_of_runs: int


class EnvironmentRunner:
    """Coordinates simulated environment lifetime.

    Spawns environment, loops over episodes.
    """
    def __init__(self, config: EnvironmentRunnerConfig):
        self._environment = EnvironmentFactory().create(config.environment_type,
                                                        config.environment_config)

