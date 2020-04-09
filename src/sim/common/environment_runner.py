
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from src.core.logger import get_logger, cprint
from src.core.config_loader import Config
from src.core.utils import get_filename_without_extension
from src.sim.common.actors import DnnActor
from src.sim.common.environment import EnvironmentConfig
from src.sim.common.data_types import TerminalType, EnvironmentType
from src.sim.environment_factory import EnvironmentFactory
from src.data.dataset_saver import DataSaver


@dataclass_json
@dataclass
class EnvironmentRunnerConfig(Config):
    environment_config: EnvironmentConfig = None
    number_of_episodes: int = None


class EnvironmentRunner:
    """Coordinates simulated environment lifetime.

    Spawns environment, loops over episodes, loops over steps in episodes.
    """
    def __init__(self, config: EnvironmentRunnerConfig, data_saver: DataSaver = None):
        self._config = config
        self._logger = get_logger(name=get_filename_without_extension(__file__),
                                  output_path=config.output_path,
                                  quite=False)

        self._data_saver = data_saver
        self._environment = EnvironmentFactory().create(config.environment_config)
        # actor is not used for ros-gazebo environments.
        self._actor = None if config.environment_config.factory_key == EnvironmentType.Ros \
            else DnnActor(config=self._config.environment_config.actor_config)
        cprint('initiated', self._logger)

    def _run_episode(self):
        state = self._environment.reset()
        while state.terminal == TerminalType.Unknown:
            state = self._environment.step()
        cprint(f'environment is running', self._logger)
        while state.terminal == TerminalType.NotDone:
            action = self._actor.get_action(state.sensor_data) if self._actor is not None else None
            state = self._environment.step(action)  # action is not used for ros-gazebo environments.
            if self._data_saver is not None:
                self._data_saver.save(state=state,
                                      action=action)
        cprint(f'environment is terminated with {state.terminal.name}', self._logger)
        if self._data_saver is not None:
            self._data_saver.save(state=state)

    def run(self):
        for self._run_index in range(self._config.number_of_episodes):
            cprint(f'start episode {self._run_index}', self._logger)
            if self._run_index > 0:
                self._data_saver.update_saving_directory()
            self._run_episode()
        if self._data_saver is not None:
            self._data_saver.create_train_validation_hdf5_files()

    def shutdown(self) -> bool:
        return self._environment.remove()
