#!/usr/bin/python3.7
import os

from dataclasses import dataclass
from dataclasses_json import dataclass_json
from torch.utils.tensorboard import SummaryWriter

from src.ai.base_net import ArchitectureConfig
from src.ai.evaluator import EvaluatorConfig, Evaluator
from src.ai.trainer import TrainerConfig, Trainer
from src.ai.architectures import *  # Do not remove
from src.core.utils import get_date_time_tag, get_filename_without_extension
from src.core.data_types import TerminationType, Distribution, Action
from src.core.config_loader import Config, Parser
from src.core.logger import get_logger, cprint, MessageType
from src.data.data_saver import DataSaverConfig, DataSaver
from src.sim.common.environment import EnvironmentConfig
from src.sim.common.environment_factory import EnvironmentFactory

"""Script for collecting dataset / evaluating a model in simulated or real environment.

Script starts environment runner with dataset_saver object to store the episodes in a dataset.
"""


@dataclass_json
@dataclass
class ExperimentConfig(Config):
    number_of_epochs: int = 1
    number_of_episodes: int = -1
    environment_config: EnvironmentConfig = None
    data_saver_config: DataSaverConfig = None
    architecture_config: ArchitectureConfig = None
    trainer_config: TrainerConfig = None
    evaluator_config: EvaluatorConfig = None
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
        self._writer = SummaryWriter(log_dir=config.output_path) \
            if self._config.tensorboard else None
        self._data_saver = DataSaver(config=config.data_saver_config) \
            if self._config.data_saver_config is not None else None
        self._environment = EnvironmentFactory().create(config.environment_config) \
            if self._config.environment_config is not None else None
        self._net = eval(config.architecture_config.architecture).Net(config=config.architecture_config) \
            if self._config.architecture_config is not None else None
        self._trainer = Trainer(config=self._config.trainer_config, network=self._net) \
            if self._config.trainer_config is not None else None
        self._evaluator = Evaluator(config=self._config.evaluator_config, network=self._net) \
            if self._config.evaluator_config is not None else None
        cprint(f'Initiated.', self._logger)

    def _run_episodes(self):
        count_episodes = 0
        while count_episodes < self._config.number_of_episodes or self._config.number_of_episodes == -1:
            cprint(f'running episode {count_episodes}', self._logger)
            experience = self._environment.reset()
            while experience.done == TerminationType.Unknown:
                experience = self._environment.step()
            while experience.done == TerminationType.NotDone:
                action = self._net.get_action(inputs=experience.observation,
                                              train=False) if self._net is not None else None
                experience = self._environment.step(action)  # action is not used for ros-gazebo environments off-policy
                if self._data_saver is not None:
                    self._data_saver.save(experience=experience)
            cprint(f'environment is terminated with {experience.done.name}', self._logger)
            count_episodes += 1
        if self._data_saver is not None:
            self._data_saver.create_train_validation_hdf5_files()

    def _log_loss(self, name: str, data: Distribution) -> str:
        if self._writer is not None:
            self._writer.add_scalar(f"{name} mean", data.mean, global_step=self._net.global_step)
            self._writer.add_scalar(f"{name} std", data.std, global_step=self._net.global_step)
        return f' {name} {data.mean: 0.3e} [{data.std: 0.3e}]'

    def run(self):
        for epoch in range(self._config.number_of_epochs):
            msg = f'epoch: {epoch + 1} / {self._config.number_of_epochs}'
            if self._environment is not None:
                self._run_episodes()
            if self._trainer is not None:
                training_loss_distribution = self._trainer.train(epoch=epoch)  # include checkpoint saving.
                msg += self._log_loss('training', training_loss_distribution)
            if self._evaluator is not None:  # if validation error is minimal then save best checkpoint
                validation_error_distribution = self._evaluator.evaluate(save_checkpoints=self._trainer is not None)
                msg += self._log_loss('validation', validation_error_distribution)
            cprint(msg, self._logger)
        cprint(f'Finished.', self._logger)

    def shutdown(self):
        if self._writer is not None:
            self._writer.close()
        if self._environment is not None:
            result = self._environment.remove()
            cprint(f'Terminated successfully? {result}', self._logger,
                   msg_type=MessageType.info if result else MessageType.warning)


if __name__ == "__main__":
    config_file = Parser().parse_args().config
    experiment_config = ExperimentConfig().create(config_file=config_file)
    experiment = Experiment(experiment_config)
    experiment.run()
    experiment.shutdown()
