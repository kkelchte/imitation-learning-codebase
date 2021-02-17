"""
Interact with environment, looping over episodes till ending criteria is met
"""
from typing import Tuple

from dataclasses import dataclass
import numpy as np
from dataclasses_json import dataclass_json

from src.ai.base_net import BaseNet
from src.core.config_loader import Config
from src.core.data_types import Distribution, TerminationType
from src.core.logger import get_logger
from src.core.utils import get_filename_without_extension
from src.data.data_saver import DataSaver
from src.sim.common.environment import Environment


@dataclass_json
@dataclass
class EpisodeRunnerConfig(Config):
    number_of_episodes: int = -1
    train_every_n_steps: int = -1
    number_of_test_episodes: int = 10

    def __post_init__(self):
        assert not (self.number_of_episodes != -1 and self.train_every_n_steps != -1), \
            'Should specify either number of episodes per epoch or number of steps before training.'


class EpisodeRunner:

    def __init__(self, config: EpisodeRunnerConfig, data_saver: DataSaver = None,
                 environment: Environment = None, net: BaseNet = None, writer=None):
        self._config = config
        self._logger = get_logger(name=get_filename_without_extension(__file__),
                                  output_path=config.output_path,
                                  quiet=False)
        self._data_saver = data_saver
        self._environment = environment
        self._net = net
        self._writer = writer
        self._max_mean_return = None
        self._min_mean_return = None
        self._reset()

    def _enough_episodes_check(self, episode_number: int, test: bool = False) -> bool:
        if not test:
            if self._config.number_of_episodes != -1:
                return episode_number >= self._config.number_of_episodes
            elif self._config.train_every_n_steps != -1:
                return len(self._data_saver) >= self._config.train_every_n_steps
            else:
                raise NotImplementedError
        else:
            if self._config.number_of_test_episodes != -1:
                return episode_number >= self._config.number_of_test_episodes
            else:
                raise NotImplementedError

    def _reset(self):
        self._frames = []
        self._count_episodes = 0
        self._count_success = 0
        self._episode_returns = []
        self._episode_lengths = []

    def _run_episode(self, store_frames: bool = False, test: bool = False, agent_id: int = -1) -> int:
        count_steps = 0
        _episode_return = 0
        experience, next_observation = self._environment.reset()
        while experience.done == TerminationType.NotDone and not self._enough_episodes_check(self._count_episodes,
                                                                                             test=test):
            kwargs = {'inputs': next_observation}
            if agent_id != -1:
                kwargs['agent_id'] = agent_id
            action = self._net.get_action(**kwargs) if self._net is not None else None
            experience, next_observation = self._environment.step(action)
            _episode_return += experience.info['unfiltered_reward'] \
                if 'unfiltered_reward' in experience.info.keys() else experience.reward
            if 'frame' in experience.info.keys() and store_frames:
                self._frames.append(experience.info['frame'])
            if self._data_saver is not None and not test:
                self._data_saver.save(experience=experience)
            count_steps += 1
        self._episode_lengths.append(count_steps)
        self._count_success += 1 if experience.done.name == TerminationType.Success.name else 0
        self._count_episodes += 1
        self._episode_returns.append(experience.info['return'] if 'return' in experience.info.keys()
                                     else _episode_return)
        return count_steps

    def _get_result_message(self, test: bool = False, tag: str = '', lowest_return: bool = False):
        msg = f" {'' if not test else 'test'} {tag} "
        msg += f"{self._count_episodes} episodes"
        if self._count_success != 0:
            msg += f" with {self._count_success} success"
            if self._writer is not None:
                self._writer.write_scalar(self._count_success / float(self._count_episodes), "success")
        return_distribution = Distribution(self._episode_returns)
        msg += f" with return {return_distribution.mean: 0.3e} [{return_distribution.std: 0.2e}]"
        if self._writer is not None:
            self._writer.write_scalar(np.mean(self._episode_lengths).item(),
                                      f'{"" if not test else "test_"}episode_lengths{"_"+tag if tag != "" else ""}')
            self._writer.write_distribution(return_distribution,
                                            f'{"" if not test else "test_"}episode_return{"_"+tag if tag != "" else ""}')
            self._writer.write_gif(self._frames,
                                   f'{"" if not test else "test_"}episode{"_"+tag if tag != "" else ""}')

        best_checkpoint = False
        if self._max_mean_return is None or return_distribution.mean > self._max_mean_return:
            self._max_mean_return = return_distribution.mean
            best_checkpoint = True
        if self._min_mean_return is None or return_distribution.mean < self._min_mean_return:
            self._min_mean_return = return_distribution.mean
            if lowest_return:
                best_checkpoint = True
        return msg, best_checkpoint

    def run_adversarial_test(self, store_frames: bool = False) -> Tuple[str, bool]:
        """
        Run test episodes for two parts of the adversarial network
        :param store_frames: bool whether a gif should be created of episodes
        :return: result message and whether current checkpoint is best
        """
        self._reset()
        while not self._enough_episodes_check(self._count_episodes, test=True):
            self._run_episode(store_frames=store_frames,
                              test=True,
                              agent_id=0)
        msg0, best_checkpoint0 = self._get_result_message(test=True,
                                                          tag='agent_0',)
        self._reset()
        while not self._enough_episodes_check(self._count_episodes, test=True):
            self._run_episode(store_frames=store_frames,
                              test=True,
                              agent_id=1)
        msg1, best_checkpoint1 = self._get_result_message(test=True,
                                                          tag='agent_1',
                                                          lowest_return=True)
        return msg0 + msg1, best_checkpoint0 and best_checkpoint1

    def run(self, store_frames: bool = False, test: bool = False, adversarial: bool = False) -> Tuple[str, bool]:
        """
        Run episodes interactively with environment up until enough data is gathered.
        :return: output message for epoch string, whether current model is the best so far.
        """
        if test and adversarial:
            return self.run_adversarial_test(store_frames=store_frames)
        self._reset()
        while not self._enough_episodes_check(self._count_episodes, test=test):
            if self._data_saver is not None and not test:
                self._data_saver.update_saving_directory()
            self._run_episode(store_frames=store_frames,
                              test=test)
        return self._get_result_message(test)

    def remove(self):
        [h.close() for h in self._logger.handlers]
