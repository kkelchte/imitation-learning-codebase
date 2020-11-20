#!/usr/bin/python
import os
import time
from glob import glob
import shutil
from typing import Optional, Tuple

import torch
import yaml
import numpy as np
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from src.ai.base_net import ArchitectureConfig
from src.ai.evaluator import EvaluatorConfig, Evaluator
from src.ai.trainer import TrainerConfig
from src.ai.architectures import *  # Do not remove
from src.ai.trainer_factory import TrainerFactory
from src.core.utils import get_filename_without_extension, get_data_dir, get_date_time_tag
from src.core.data_types import TerminationType, Distribution, Dataset
from src.core.config_loader import Config, Parser
from src.core.logger import get_logger, cprint, MessageType
from src.data.data_saver import DataSaverConfig, DataSaver
from src.scripts.episode_runner import EpisodeRunnerConfig, EpisodeRunner
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
    episode_runner_config: Optional[EpisodeRunnerConfig] = None
    load_checkpoint_dir: Optional[str] = None  # path to checkpoints
    load_checkpoint_found: bool = True
    save_checkpoint_every_n: int = -1
    tensorboard: bool = False
    tb_render_every_n_epochs: int = -1
    environment_config: Optional[EnvironmentConfig] = None
    data_saver_config: Optional[DataSaverConfig] = None
    architecture_config: Optional[ArchitectureConfig] = None
    trainer_config: Optional[TrainerConfig] = None
    evaluator_config: Optional[EvaluatorConfig] = None
    tester_config: Optional[EvaluatorConfig] = None
    run_test_episodes: bool = False

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
        if self.tester_config is None:
            del self.tester_config
        if self.load_checkpoint_dir is None:
            del self.load_checkpoint_dir
        if self.episode_runner_config is None:
            del self.episode_runner_config
        # if episode runner config has no limit on duration, use trainers batch size
        if self.episode_runner_config is not None and \
                self.episode_runner_config.number_of_episodes == -1 and \
                self.episode_runner_config.train_every_n_steps == -1 and \
                self.trainer_config is not None:
            self.episode_runner_config.train_every_n_steps = self.trainer_config.data_loader_config.batch_size


class Experiment:

    def __init__(self, config: ExperimentConfig):
        np.random.seed(123)
        self._epoch = 0
        self._config = config
        self._logger = get_logger(name=get_filename_without_extension(__file__),
                                  output_path=config.output_path,
                                  quiet=False)
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
        self._tester = None  # create at the end to avoid too much data is loaded in RAM
        self._writer = None
        if self._config.tensorboard:  # Local import so code can run without tensorboard
            from src.core.tensorboard_wrapper import TensorboardWrapper
            self._writer = TensorboardWrapper(log_dir=config.output_path)
        self._episode_runner = EpisodeRunner(config=self._config.episode_runner_config,
                                             data_saver=self._data_saver,
                                             environment=self._environment,
                                             net=self._net,
                                             writer=self._writer) \
            if self._config.episode_runner_config is not None else None

        if self._config.load_checkpoint_found \
                and len(glob(f'{self._config.output_path}/torch_checkpoints/*.ckpt')) > 0:
            self.load_checkpoint(f'{self._config.output_path}/torch_checkpoints')
        elif self._config.load_checkpoint_dir is not None:
            if not self._config.load_checkpoint_dir.startswith('/'):
                self._config.load_checkpoint_dir = f'{get_data_dir(self._config.output_path)}/' \
                                                   f'{self._config.load_checkpoint_dir}'
            self.load_checkpoint(self._config.load_checkpoint_dir)

        cprint(f'Initiated.', self._logger)

    def run(self):
        for self._epoch in range(self._config.number_of_epochs):
            best_ckpt = False
            msg = f'{get_date_time_tag()} epoch: {self._epoch + 1} / {self._config.number_of_epochs}'
            if self._environment is not None:
                if self._data_saver is not None and self._config.data_saver_config.clear_buffer_before_episode:
                    self._data_saver.clear_buffer()
                output_msg, best_ckpt = self._episode_runner.run(
                    store_frames=(self._config.tb_render_every_n_epochs != -1 and
                                  self._epoch % self._config.tb_render_every_n_epochs == 0 and
                                  self._writer is not None))
                if self._data_saver is not None and self._config.data_saver_config.store_hdf5:
                    self._data_saver.create_train_validation_hdf5_files()
                msg += output_msg
            if self._trainer is not None:
                if self._data_saver is not None:  # update fresh data to train
                    self._trainer.data_loader.set_dataset(
                        self._data_saver.get_dataset() if self._config.data_saver_config.store_on_ram_only else None
                    )
                msg += self._trainer.train(epoch=self._epoch, writer=self._writer)
            if self._evaluator is not None:  # if validation error is minimal then save best checkpoint
                output_msg, best_ckpt = self._evaluator.evaluate(epoch=self._epoch, writer=self._writer)
                msg += output_msg
            if self._config.run_test_episodes:
                output_msg, best_ckpt = self._episode_runner.run(
                    store_frames=(self._config.tb_render_every_n_epochs != -1 and
                                  self._epoch % self._config.tb_render_every_n_epochs == 0 and
                                  self._writer is not None),
                    test=True,
                    # ! adversarial tag should be in architecture name
                    adversarial='adversarial' in self._config.architecture_config.architecture
                )
                msg += output_msg
            if self._config.save_checkpoint_every_n != -1 and \
                    (self._epoch % self._config.save_checkpoint_every_n == 0 or
                     self._epoch == self._config.number_of_epochs - 1):
                self.save_checkpoint(tag=f'{self._epoch:05d}')
            if best_ckpt and self._config.save_checkpoint_every_n != -1:
                self.save_checkpoint(tag='best')
            cprint(msg, self._logger)
        if self._trainer is not None:
            self._trainer.data_loader.empty_dataset()
        if self._evaluator is not None and self._config.evaluator_config.evaluate_extensive:
            self._evaluator.evaluate_extensive()
        if self._evaluator is not None:
            self._evaluator.data_loader.empty_dataset()
        self._tester = Evaluator(config=self._config.tester_config, network=self._net) \
            if self._config.tester_config is not None else None
        if self._tester is not None:
            output_msg, _ = self._tester.evaluate(epoch=self._epoch, writer=self._writer, tag='test')
            cprint(f'Testing: {output_msg}', self._logger)
            if self._config.tester_config.evaluate_extensive:
                self._tester.evaluate_extensive()

        cprint(f'Finished.', self._logger)

    def save_checkpoint(self, tag: str = ''):
        filename = f'checkpoint_{tag}' if tag != '' else 'checkpoint'
        filename += '.ckpt'
        checkpoint = {
            'epoch': self._epoch,
        }
        for element, key in zip([self._net, self._trainer, self._environment],
                                ['net_ckpt', 'trainer_ckpt', 'environment_ckpt']):
            if element is not None:
                checkpoint[key] = element.get_checkpoint()
        os.makedirs(f'{self._config.output_path}/torch_checkpoints', exist_ok=True)
        torch.save(checkpoint, f'{self._config.output_path}/torch_checkpoints/{filename}')
        torch.save(checkpoint, f'{self._config.output_path}/torch_checkpoints/checkpoint_latest.ckpt')
        cprint(f'stored {filename}', self._logger)

    def load_checkpoint(self, checkpoint_dir: str):
        if not checkpoint_dir.endswith('torch_checkpoints'):
            checkpoint_dir += '/torch_checkpoints'
        if len(glob(f'{checkpoint_dir}/*.ckpt')) == 0 and len(glob(f'{checkpoint_dir}/torch_checkpoints/*.ckpt')) == 0:
            cprint(f'Could not find suitable checkpoint in {checkpoint_dir}', self._logger, MessageType.error)
            time.sleep(0.1)
            raise FileNotFoundError
        # Get checkpoint in following order
        if os.path.isfile(os.path.join(checkpoint_dir, 'checkpoint_best.ckpt')):
            checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint_best.ckpt')
        elif os.path.isfile(os.path.join(checkpoint_dir, 'checkpoint_latest.ckpt')):
            checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint_latest.ckpt')
        else:
            checkpoints = {int(f.split('.')[0].split('_')[-1]): os.path.join(checkpoint_dir, f)
                           for f in os.listdir(checkpoint_dir)}
            checkpoint_file = checkpoints[max(checkpoints.keys())]
        # Load params for each experiment element
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        self._epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else 0
        for element, key in zip([self._net, self._trainer, self._environment],
                                ['net_ckpt', 'trainer_ckpt', 'environment_ckpt']):
            if element is not None and key in checkpoint.keys():
                element.load_checkpoint(checkpoint[key])
        cprint(f'loaded network from {checkpoint_file}', self._logger)

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
        if self._episode_runner is not None:
            self._episode_runner.remove()
        [h.close() for h in self._logger.handlers]


if __name__ == "__main__":
    arguments = Parser().parse_args()
    config_file = arguments.config
    if arguments.rm:
        with open(config_file, 'r') as f:
            configuration = yaml.load(f, Loader=yaml.FullLoader)
        if not configuration['output_path'].startswith('/'):
            configuration['output_path'] = os.path.join(get_data_dir(os.environ['HOME']), configuration['output_path'])
        shutil.rmtree(configuration['output_path'], ignore_errors=True)

    experiment_config = ExperimentConfig().create(config_file=config_file,
                                                  seed=arguments.seed)
    experiment = Experiment(experiment_config)
    experiment.run()
    experiment.shutdown()
