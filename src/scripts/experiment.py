#!/usr/bin/python3.7
import os
from typing import Optional

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from src.ai.base_net import ArchitectureConfig
from src.ai.evaluator import EvaluatorConfig, Evaluator
from src.ai.trainer import TrainerConfig, Trainer
from src.ai.architectures import *  # Do not remove
from src.ai.trainer_factory import TrainerFactory
from src.core.utils import get_date_time_tag, get_filename_without_extension
from src.core.data_types import TerminationType, Distribution, Action
from src.core.config_loader import Config, Parser
from src.core.logger import get_logger, cprint, MessageType
from src.data.data_saver import DataSaverConfig, DataSaver
from src.sim.common.environment import EnvironmentConfig
from src.sim.common.environment_factory import EnvironmentFactory


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
"""Script for collecting dataset / evaluating a model in simulated or real environment.

Script starts environment runner with dataset_saver object to store the episodes in a dataset.
"""


@dataclass_json
@dataclass
class ExperimentConfig(Config):
    number_of_epochs: int = 1
    number_of_episodes: int = -1
    environment_config: Optional[EnvironmentConfig] = None
    data_saver_config: Optional[DataSaverConfig] = None
    architecture_config: Optional[ArchitectureConfig] = None
    trainer_config: Optional[TrainerConfig] = None
    evaluator_config: Optional[EvaluatorConfig] = None
    tensorboard: bool = False

    def __post_init__(self):
        # Avoid None value error by deleting irrelevant fields
        if self.environment_config is None:
            del self.environment_config
        if self.data_saver_config is None:
            del self.data_saver_config
        if self.architecture_config is None:
            del self.architecture_config
        if self.trainer_config is None:
            del self.trainer_config
        if self.evaluator_config is None:
            del self.evaluator_config


class Experiment:

    def __init__(self, config: ExperimentConfig):
        self._config = config
        self._logger = get_logger(name=get_filename_without_extension(__file__),
                                  output_path=config.output_path,
                                  quite=False)
        self._data_saver = DataSaver(config=config.data_saver_config) \
            if self._config.data_saver_config is not None else None
        self._environment = EnvironmentFactory().create(config.environment_config) \
            if self._config.environment_config is not None else None
        self._net = eval(config.architecture_config.architecture).Net(config=config.architecture_config) \
            if self._config.architecture_config is not None else None
        self._trainer = TrainerFactory().create(config=self._config.trainer_config, network=self._net) \
            if self._config.trainer_config is not None else None
        self._evaluator = Evaluator(config=self._config.evaluator_config, network=self._net) \
            if self._config.evaluator_config is not None else None
        self._writer = None
        if self._config.tensorboard:  # Local import so code can run without tensorboard
            from src.core.tensorboard_wrapper import TensorboardWrapper
            self._writer = TensorboardWrapper(log_dir=config.output_path)
        cprint(f'Initiated.', self._logger)

    def _enough_episodes_check(self, episode_number: int) -> bool:
        if self._config.number_of_episodes != -1:
            return episode_number >= self._config.number_of_episodes
        else:
            #  TODO check data save dataset length vs trainer's data loader batch size
            #  TODO add prefill option in first epoch
            raise NotImplementedError

    def _run_episodes(self) -> str:
        count_episodes = 0
        while not self._enough_episodes_check(count_episodes):
            experience = self._environment.reset()
            while experience.done == TerminationType.Unknown:
                experience = self._environment.step()
            while experience.done == TerminationType.NotDone:
                action = self._net.get_action(inputs=experience.observation,
                                              train=False) if self._net is not None else None
                cprint(action, self._logger)
                experience = self._environment.step(action)  # action is not used for ros-gazebo environments off-policy
                if self._data_saver is not None:
                    self._data_saver.save(experience=experience)
            # cprint(f'environment is terminated episode {count_episodes} with {experience.done.name}', self._logger)
            count_episodes += 1
        if self._data_saver is not None:
            self._data_saver.create_train_validation_hdf5_files()
        return f" {count_episodes} episodes"

    def run(self):
        for self._epoch in range(self._config.number_of_epochs):
            msg = f'epoch: {self._epoch + 1} / {self._config.number_of_epochs}'
            if self._environment is not None:
                msg += self._run_episodes()
            if self._trainer is not None:
                if self._data_saver is not None:  # update fresh data to train
                    self._trainer.data_loader.set_dataset(
                        self._data_saver.get_dataset() if self._config.data_saver_config.store_on_ram_only else None
                    )
                msg += self._trainer.train(epoch=self._epoch,
                                           writer=self._writer)
            if self._evaluator is not None:  # if validation error is minimal then save best checkpoint
                msg += self._evaluator.evaluate(save_checkpoints=self._trainer is not None,
                                                writer=self._writer)
            cprint(msg, self._logger)
        cprint(f'Finished.', self._logger)

    def shutdown(self):
        if self._writer is not None:
            self._writer.close()
        if self._environment is not None:
            result = self._environment.remove()
            cprint(f'Terminated successfully? {bool(result)}', self._logger,
                   msg_type=MessageType.info if result else MessageType.warning)
        if self._data_saver is not None:
            self._data_saver.remove()
        if self._trainer is not None:
            self._trainer.remove()
        if self._evaluator is not None:
            self._evaluator.remove()
        if self._net is not None:
            self._net.remove()
        [h.close() for h in self._logger.handlers]


if __name__ == "__main__":
    config_file = Parser().parse_args().config
    experiment_config = ExperimentConfig().create(config_file=config_file)
    experiment = Experiment(experiment_config)
    experiment.run()
    experiment.shutdown()
