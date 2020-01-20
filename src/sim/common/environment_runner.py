
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from src.core.logger import get_logger
from src.core.config_loader import Config
from src.sim.common.environment import EnvironmentConfig
from src.sim.common.data_types import ActorType, TerminalType
from src.sim.common.actors import DNNActor, ActorConfig
from src.sim.environment_factory import EnvironmentFactory
from src.data.dataset_saver import DataSaver


@dataclass_json
@dataclass
class EnvironmentRunnerConfig(Config):
    environment_config: EnvironmentConfig = None
    actor_config: ActorConfig = None
    number_of_episodes: int = None


class EnvironmentRunner:
    """Coordinates simulated environment lifetime.

    Spawns environment, loops over episodes, loops over steps in episodes.
    """
    def __init__(self, config: EnvironmentRunnerConfig, data_saver: DataSaver = None):
        self._config = config
        logger = get_logger(name=__name__,
                            output_path=config.output_path,
                            quite=False)
        logger.info(f'Initiate.')
        self._data_saver = data_saver
        self._environment = EnvironmentFactory().create(config.environment_config)
        if self._config.actor_config.actor_type == ActorType.Model:
            self._actor = DNNActor(config=config.actor_config)
        else:
            self._actor = self._environment.get_actor(actor_config=self._config.actor_config)

    def _run_episode(self):
        state = self._environment.reset()
        while state.terminal == TerminalType.NotDone:
            action = self._actor(state.sensor_data)
            state = self._environment.step(action)
            if self._data_saver is not None:
                self._data_saver.save(state=state,
                                      action=action)
        if self._data_saver is not None:
            self._data_saver.save(state=state)

    def run(self):
        for self._run_index in range(self._config.number_of_episodes):
            self._run_episode()
