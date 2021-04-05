import os
from enum import IntEnum

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np

from dataclasses_json import dataclass_json

from src.core.config_loader import Config
from src.core.filters import NormalizationFilter, ReturnFilter
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension
from src.core.data_types import Action, Experience, ProcessState, SensorType
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
    # defines world params and if gazebo world to spaw
    world_name: str = 'default'
    # selects driver node and robot params: turtlebot_sim, turtlebot_real, bebop_real, drone_sim, ...
    robot_name: str = None
    gazebo: bool = False
    robot_display: bool = False
    fsm: bool = True
    fsm_mode: str = 'SingleRun'
    robot_mapping: bool = False  # map trajectory on top down map specified by 'background image' in world config
    control_mapping: bool = True
    control_mapping_config: str = 'default'
    modified_state_publisher: bool = False
    modified_state_publisher_mode: str = 'CombinedGlobalPoses'
    modified_state_frame_visualizer: bool = False
    waypoint_indicator: bool = False  # configuration is specified in world
    x_pos: float = 0.
    y_pos: float = 0.
    z_pos: float = 0.
    yaw_or: float = 0.
    starting_height: float = 1.
    starting_height_tracking: float = 1.
    starting_height_fleeing: float = 1.
    distance_tracking_fleeing_m: float = 3.
    altitude_control: bool = False

    def __post_init__(self):
        assert os.path.isfile(f'src/sim/ros/config/control_mapping/{self.control_mapping_config}.yml')


@dataclass_json
@dataclass
class RosConfig(Config):
    """
    Configuration specific for ROS environment,
    specified here to avoid circular dependencies environment <> ros_environment
    """
    # sensor_type which has to be specified by robot
    observation: str = 'camera'
    # control_topic which corresponds to the published action or python then it returns action from step argument
    action_topic: str = '/cmd_vel'
    # sensor_type_0, current_waypoint, actor_topic_0
    info: Optional[List[Union[SensorType, str]]] = None
    step_rate_fps: float = 10.
    visible_xterm: bool = False  # make xterm window in which launch load_ros is launched visible
    ros_launch_config: RosLaunchConfig = None
    actor_configs: Optional[List[ActorConfig]] = None
    # list of all actors that will be started by load_ros.launch + config files
    max_update_wait_period_s: float = 10  # max wall time waiting duration till update.
    # define number of action publishers used by adversarial or multi-agent training
    # result in topics: /ros_python_interface/cmd_vel, /ros_python_interface/cmd_vel_1 ...
    num_action_publishers: int = 1

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
