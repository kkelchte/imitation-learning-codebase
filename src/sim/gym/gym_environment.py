import gym
import numpy as np
from copy import deepcopy

from src.sim.common.data_types import State, TerminalType, Action, ActorType
from src.sim.common.environment import EnvironmentConfig, Environment


class GymEnvironment(Environment):

    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)
        self._gym = gym.make(self._config.gym_config.world_name)
        self.discrete = isinstance(self._gym.action_space, gym.spaces.Discrete)
        self.previous_observation = None
        self.observation_dimension = self._gym.observation_space.shape[0]
        self.action_dimension = self._gym.action_space.n if self.discrete else self._gym.action_space.shape[0]

    def reset(self) -> State:
        observation = self._gym.reset()
        self.previous_observation = observation
        return State(
            terminal=TerminalType.NotDone,
            actor_data={},
            sensor_data={
                'observation': deepcopy(self.previous_observation)
            }
        )

    def step(self, action: Action) -> State:
        observation, reward, done, info = self._gym.step(action.value)
        state = State(
            terminal=TerminalType.NotDone if not done else TerminalType.Done,
            actor_data={'dnn_actor': action},
            sensor_data={
                'observation': deepcopy(self.previous_observation),
                'next_observation': deepcopy(observation),
                'reward': reward,
                'info': info
            }
        )
        self.previous_observation = observation
        return state

    def remove(self):
        self._gym.close()
