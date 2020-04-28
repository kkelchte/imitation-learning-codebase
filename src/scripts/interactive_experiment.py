
"""Script for collecting dataset / evaluating a model in simulated or real environment.

Script starts environment runner with dataset_saver object to store the episodes in a dataset.
"""

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from src.core.utils import get_filename_without_extension
from src.sim.ros.catkin_ws.src.imitation_learning_ros_package.rosnodes.actors import DnnActor
from src.sim.common.environment import EnvironmentConfig
from src.core.data_types import TerminationType, EnvironmentType
from src.sim.environment_factory import EnvironmentFactory
from src.core.config_loader import Config, Parser
from src.core.logger import get_logger, cprint, MessageType
from src.data.dataset_saver import DataSaverConfig, DataSaver


@dataclass_json
@dataclass
class InteractiveExperimentConfig(Config):
    data_saver_config: DataSaverConfig = None
    environment_config: EnvironmentConfig = None
    number_of_episodes: int = None


class InteractiveExperiment:
    """Coordinates simulated environment lifetime.

    Spawns environment, loops over episodes, loops over steps in episodes.
    """
    def __init__(self, config: InteractiveExperimentConfig):
        self._config = config
        self._logger = get_logger(name=get_filename_without_extension(__file__),
                                  output_path=config.output_path,
                                  quite=False)

        self._data_saver = DataSaver(config=config.data_saver_config)
        self._environment = EnvironmentFactory().create(config.environment_config)
        # actor is not used for ros-gazebo environments.
        self._actor = None if config.environment_config.factory_key == EnvironmentType.Ros \
            else DnnActor(config=self._config.environment_config.actor_config)
        cprint('initiated', self._logger)

    def _run_episode(self):
        experience = self._environment.reset()
        while experience.terminal == TerminationType.Unknown:
            experience = self._environment.step()
        cprint(f'environment is running', self._logger)
        while experience.terminal == TerminationType.NotDone:
            action = self._actor.get_action(experience.observation) if self._actor is not None else None
            experience = self._environment.step(action)  # action is not used for ros-gazebo environments.
            if self._data_saver is not None:
                self._data_saver.save(experience=experience)
        cprint(f'environment is terminated with {experience.terminal.name}', self._logger)
        if self._data_saver is not None:
            self._data_saver.save(experience=experience)

    def run(self):
        for self._run_index in range(self._config.number_of_episodes):
            cprint(f'start episode {self._run_index}', self._logger)
            if self._run_index > 0:
                self._data_saver.update_saving_directory()
            self._run_episode()
        if self._data_saver is not None:
            self._data_saver.create_train_validation_hdf5_files()

    def shutdown(self):
        result = self._environment.remove()
        cprint(f'Terminated successfully? {result}', self._logger,
               msg_type=MessageType.info if result else MessageType.warning)


if __name__ == "__main__":
    config_file = Parser().parse_args().config
    experiment_config = InteractiveExperimentConfig().create(config_file=config_file)
    experiment = InteractiveExperiment(experiment_config)
    experiment.run()
    experiment.shutdown()
