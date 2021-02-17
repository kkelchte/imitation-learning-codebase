import os
from enum import IntEnum

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from dataclasses_json import dataclass_json

from src.core.config_loader import Config
from src.core.filters import NormalizationFilter, ReturnFilter
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension
from src.core.data_types import Action, Experience, ProcessState
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.actors import ActorConfig


@dataclass_json
@dataclass
class GymConfig(Config):
    """
    Configuration specific for Gym environment,
    specified here to avoid circular dependencies environment <> gym_environment
    """
    random_seed: int = 123
    world_name: str = None
    render: bool = False
    args: str = ""


@dataclass_json
@dataclass
class RosLaunchConfig(Config):
    random_seed: int = 123
    gazebo: bool = False
    robot_display: bool = False
    fsm: bool = True
    fsm_config: str = 'single_run'
    control_mapping: bool = True
    control_mapping_config: str = 'default'
    waypoint_indicator: bool = True
    x_pos: float = 0.
    y_pos: float = 0.
    z_pos: float = 1.
    yaw_or: float = 1.57
    world_name: str = None
    robot_name: str = None
    robot_mapping: bool = True

    def __post_init__(self):
        assert os.path.isfile(f'src/sim/ros/config/fsm/{self.fsm_config}.yml')
        assert os.path.isfile(f'src/sim/ros/config/control_mapping/{self.control_mapping_config}.yml')


@dataclass_json
@dataclass
class RosConfig(Config):
    """
    Configuration specific for ROS environment,
    specified here to avoid circular dependencies environment <> ros_environment
    """
    store_action: bool = True
    store_reward: bool = False
    observation: str = ''
    action: str = ''
    # sensor/sensor_name_0, sensor/sensor_name_1, actor/actor_name_0, ..., current_waypoint, supervised_action
    info: Optional[List[str]] = None
    step_rate_fps: float = 10.
    visible_xterm: bool = False
    ros_launch_config: RosLaunchConfig = None
    actor_configs: List[ActorConfig] = None  # extra ros nodes that can act on robot.
    max_update_wait_period_s: float = 10  # max wall time waiting duration till update.

    def __post_init__(self):
        if self.info is None:
            del self.info
        if self.actor_configs is None:
            del self.actor_configs


@dataclass_json
@dataclass
class EnvironmentConfig(Config):
    """
    Serves as configuration for all environment types.
    Providing post-factory specific configuration classes is tricky due to the .from_dict
    dependency of dataclass_json which complains at unknown variables.
    """
    factory_key: str = None
    max_number_of_steps: int = 100
    # Gazebo specific environment settings
    ros_config: Optional[RosConfig] = None
    # Gym specific environment settings
    gym_config: Optional[GymConfig] = None
    normalize_observations: bool = False
    normalize_rewards: bool = False
    observation_clipping: int = -1
    reward_clipping: int = -1
    invert_reward: bool = False

    def __post_init__(self):
        if self.gym_config is None:
            del self.gym_config
        elif self.ros_config is None:
            del self.ros_config


class Environment:

    def __init__(self, config: EnvironmentConfig):
        self._config = config
        self._logger = get_logger(name=get_filename_without_extension(__file__),
                                  output_path=self._config.output_path,
                                  quiet=False)

        if self._config.normalize_observations:
            self._observation_filter = NormalizationFilter(clip=self._config.observation_clipping)

        if self._config.normalize_rewards:
            self._reward_filter = ReturnFilter(clip=self._config.reward_clipping,
                                               discount=0.99)

        cprint('initiated', self._logger)

    def step(self, action: Action) -> Tuple[Experience, np.ndarray]:
        pass

    def reset(self) -> Tuple[Experience, np.ndarray]:
        pass

    def remove(self) -> ProcessState:
        [h.close() for h in self._logger.handlers]
        return ProcessState.Terminated

    def _filter_observation(self, observation: np.ndarray) -> np.ndarray:
        return self._observation_filter(observation) if self._config.normalize_observations else observation

    def _filter_reward(self, reward: float) -> float:
        reward = self._reward_filter(reward) if self._config.normalize_rewards else reward
        return -reward if self._config.invert_reward else reward

    def _reset_filters(self) -> None:
        if self._config.normalize_observations:
            self._observation_filter.reset()
        if self._config.normalize_rewards:
            self._reward_filter.reset()

    def get_checkpoint(self) -> dict:
        checkpoint = {}
        if self._config.normalize_observations:
            checkpoint['observation_ckpt'] = self._observation_filter.get_checkpoint()
        if self._config.normalize_rewards:
            checkpoint['reward_ckpt'] = self._reward_filter.get_checkpoint()
        return checkpoint

    def load_checkpoint(self, checkpoint: dict) -> None:
        if self._config.normalize_observations:
            self._observation_filter.load_checkpoint(checkpoint['observation_ckpt'])
        if self._config.normalize_rewards:
            self._reward_filter.load_checkpoint(checkpoint['reward_ckpt'])
