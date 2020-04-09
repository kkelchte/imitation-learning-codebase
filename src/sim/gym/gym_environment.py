import gym
import numpy as np
from copy import deepcopy

from src.core.logger import cprint, MessageType
from src.sim.common.data_types import State, TerminalType, Action, ActorType, ProcessState
from src.sim.common.environment import EnvironmentConfig, Environment


class GymEnvironment(Environment):

    def __init__(self, config: EnvironmentConfig):
        super(GymEnvironment, self).__init__(config=config)
        self._gym = gym.make(self._config.gym_config.world_name)
        self.discrete = isinstance(self._gym.action_space, gym.spaces.Discrete)
        self.previous_observation = None
        self.observation_dimension = self._gym.observation_space
        self.action_dimension = self._gym.action_space.n if self.discrete else self._gym.action_space.shape[0]
        self.action_low = None if self.discrete else self._gym.action_space.low[0]
        self.action_high = None if self.discrete else self._gym.action_space.high[0]
        self._step_count = 0
        cprint(f'environment {self._config.gym_config.world_name}\t'
               f'action space: {"discrete" if self.discrete else "continuous"} {self.action_dimension}'
               f'{"" if self.discrete else "["+str(self.action_low)+" : "+str(self.action_high)+"]"}\t'
               f'observation space: {self.observation_dimension}', self._logger)

    def reset(self) -> State:
        observation = self._gym.reset()
        self._step_count = 0
        self.previous_observation = observation
        return State(
            terminal=TerminalType.NotDone,
            actor_data={},
            sensor_data={
                'observation': deepcopy(self.previous_observation)
            }
        )

    def step(self, action: Action) -> State:
        self._step_count += 1
        observation, reward, done, info = self._gym.step(action.value)
        terminal = TerminalType.NotDone if not done and self._step_count < self._config.max_number_of_steps \
            else TerminalType.Done
        state = State(
            terminal=terminal,
            actor_data={'dnn_actor': action},
            sensor_data={
                'done': done,
                'observation': deepcopy(self.previous_observation),
                'next_observation': deepcopy(observation),
                'reward': reward,
                'info': info
            }
        )
        if self._config.gym_config.render:
            self._gym.render()
        self.previous_observation = observation
        return state

    def get_random_action(self) -> Action:
        return Action(
            actor_type=ActorType.Unknown,
            actor_name='random',
            value=np.asarray(self._gym.action_space.sample())
        )

    def remove(self) -> ProcessState:
        self._gym.close()
        return ProcessState.Terminated
