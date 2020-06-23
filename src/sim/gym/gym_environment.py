from typing import Tuple

import gym
import numpy as np
from copy import deepcopy

from src.core.logger import cprint
from src.core.data_types import Experience, TerminationType, Action, ProcessState
from src.sim.common.environment import EnvironmentConfig, Environment


class GymEnvironment(Environment):

    def __init__(self, config: EnvironmentConfig):
        super(GymEnvironment, self).__init__(config=config)
        self._gym = gym.make(self._config.gym_config.world_name)
        self._gym.seed(self._config.gym_config.random_seed)
        self.discrete = isinstance(self._gym.action_space, gym.spaces.Discrete)
        self.previous_observation = None
        self.observation_dimension = self._gym.observation_space
        self.action_dimension = self._gym.action_space.n if self.discrete else self._gym.action_space.shape[0]
        self.action_low = None if self.discrete else self._gym.action_space.low[0]
        self.action_high = None if self.discrete else self._gym.action_space.high[0]
        self._step_count = 0
        self._return = 0
        cprint(f'environment {self._config.gym_config.world_name}\t'
               f'action space: {"discrete" if self.discrete else "continuous"} {self.action_dimension}'
               f'{"" if self.discrete else "["+str(self.action_low)+" : "+str(self.action_high)+"]"}\t'
               f'observation space: {self.observation_dimension}', self._logger)

    def reset(self) -> Tuple[Experience, np.ndarray]:
        self._reset_filters()
        observation = self._gym.reset()
        observation = self._filter_observation(observation)
        self._step_count = 0
        self._return = 0
        self.previous_observation = observation.copy()
        return Experience(
            done=TerminationType.NotDone,
        ), observation

    def step(self, action: Action) -> Tuple[Experience, np.ndarray]:
        self._step_count += 1
        observation, unfiltered_reward, done, info = self._gym.step(action.value)
        observation = self._filter_observation(observation)
        reward = self._filter_reward(unfiltered_reward)
        info['unfiltered_reward'] = unfiltered_reward
        self._return += unfiltered_reward
        terminal = TerminationType.Done if done or self._step_count >= self._config.max_number_of_steps != -1 \
            else TerminationType.NotDone
        if terminal == TerminationType.Done:
            info['return'] = self._return
        experience = Experience(
            done=terminal,
            observation=self.previous_observation.copy(),
            action=action,
            reward=reward,
            time_stamp=self._step_count,
            info=info
        )
        if self._config.gym_config.render:
            self._gym.render()
        self.previous_observation = observation.copy()
        return experience, observation.copy()

    def get_random_action(self) -> Action:
        return Action(
            actor_name='random',
            value=np.asarray(self._gym.action_space.sample())
        )

    def remove(self) -> ProcessState:
        self._gym.close()
        return super().remove()
