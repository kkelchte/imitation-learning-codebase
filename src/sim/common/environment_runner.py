
from dataclasses import dataclass

from src.sim.common.environment import EnvironmentConfig
from src.sim.common.data_types import ActorType, OutcomeType
from src.sim.common.actors import DNNActor, ActorConfig
from src.sim.environment_factory import EnvironmentFactory


@dataclass
class EnvironmentRunnerConfig:
    environment_config: EnvironmentConfig = None
    actor_config: ActorConfig = None
    number_of_episodes: int = None


class EnvironmentRunner:
    """Coordinates simulated environment lifetime.

    Spawns environment, loops over episodes, loops over steps in episodes.
    """
    def __init__(self, config: EnvironmentRunnerConfig):
        self._config = config
        self._environment = EnvironmentFactory().create(config.environment_config.environment_type,
                                                        config.environment_config)
        if self._config.actor_config.actor_type == ActorType.Model:
            self._actor = DNNActor(config=config.actor_config)
        else:
            self._actor = self._environment.get_actor(actor_config=self._config.actor_config)

    def _run_episode(self):
        state = self._environment.reset()
        while state.outcome == OutcomeType.NotDone:
            action = self._actor(state)
            state = self._environment.step(action)

    def run(self):
        for self._run_index in range(self._config.number_of_episodes):
            self._run_episode()
