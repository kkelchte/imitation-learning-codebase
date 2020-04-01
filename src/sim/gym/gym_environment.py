import gym
import numpy as np

from src.sim.common.data_types import State, TerminalType, Action, ActorType
from src.sim.common.environment import EnvironmentConfig, Environment


class GymEnvironment(Environment):

    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)
        self._gym = gym.make(self._config.gym_config.world_name)

    def reset(self) -> State:
        observation = self._gym.reset()
        return State(
            terminal=TerminalType.NotDone,
            actor_data={'dnn_actor': Action(
                actor_name='dnn_actor',
                actor_type=ActorType.Model,
                value=np.zeros((3, 1))
            )},
            sensor_data={
                'observation': observation
            }
        )

    def _get_terminal_type(self, reward: float, done: bool) -> TerminalType:
        if not done:
            return TerminalType.NotDone
        else:
            if reward > 0:
                return TerminalType.Success
            else:
                return TerminalType.Failure

    def step(self, action: Action) -> State:
        observation, reward, done, info = self._gym.step(action.value)
        return State(
            terminal=self._get_terminal_type(reward, done),
            actor_data={'dnn_actor': Action(
                actor_name='dnn_actor',
                actor_type=ActorType.Model,
                value=action.value
            )},
            sensor_data={
                'observation': observation,
                'reward': reward,
                'info': info
            }
        )

    def remove(self):
        self._gym.close()
