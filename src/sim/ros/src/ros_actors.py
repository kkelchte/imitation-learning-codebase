"""All instances, besides the common instances, which can act in Gazebo environments.

Define:
- User actor
- Expert actor based on extra simulated sensors
"""
import numpy as np

from src.sim.common.actors import Actor
from src.sim.common.data_types import Action, ActorType


class RosExpert(Actor):

    def __init__(self, config):
        super().__init__(config=config)
        self._sensor_name = self._config

    def get_action(self, sensor_data: dict):
        assert self._sensor_name in sensor_data.keys(), \
            f'[ros_actors]: Failed to find {self._sensor_name} in sensor_data: {sensor_data}'
        return Action(
            actor_type=ActorType.Expert,
            value=self._calculate_action(sensor_data[self._sensor_name])
        )

    def _calculate_action(self, data: np.ndarray) -> np.ndarray:
        pass