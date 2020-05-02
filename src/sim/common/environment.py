import os
from enum import IntEnum

from dataclasses import dataclass
from typing import List

from dataclasses_json import dataclass_json

from src.core.config_loader import Config
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension
from src.core.data_types import Action, Experience, ProcessState
from src.sim.ros.catkin_ws.src.imitation_learning_ros_package.rosnodes.actors import ActorConfig


class EnvironmentType(IntEnum):
    Ros = 0
    Gym = 1
    Real = 2


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


@dataclass_json
@dataclass
class RosLaunchConfig(Config):
    random_seed: int = 123
    gazebo: bool = False
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
    observation: str = ''
    # sensor/sensor_name_0, sensor/sensor_name_1, actor/actor_name_0, ..., current_waypoint, supervised_action
    info: List[str] = None
    step_rate_fps: float = 10.
    visible_xterm: bool = False
    ros_launch_config: RosLaunchConfig = None
    actor_configs: List[ActorConfig] = None  # extra ros nodes that can act on robot.

    def __post_init__(self):
        if self.info is None:
            del self.info


@dataclass_json
@dataclass
class EnvironmentConfig(Config):
    """
    Serves as configuration for all environment types.
    Providing post-factory specific configuration classes is tricky due to the .from_dict
    dependency of dataclass_json which complains at unknown variables.
    """
    factory_key: EnvironmentType = None
    max_number_of_steps: int = 100
    # Gazebo specific environment settings
    ros_config: RosConfig = None
    # Gym specific environment settings
    gym_config: GymConfig = None

    def __post_init__(self):
        # Avoid None value error by deleting irrelevant fields
        if self.factory_key == EnvironmentType.Ros:
            del self.gym_config
        elif self.factory_key == EnvironmentType.Gym:
            del self.ros_config


class Environment:

    def __init__(self, config: EnvironmentConfig):
        self._config = config
        self._logger = get_logger(name=get_filename_without_extension(__file__),
                                  output_path=self._config.output_path,
                                  quite=False)
        cprint('initiated', self._logger)

    def step(self, action: Action) -> Experience:
        pass

    def reset(self) -> Experience:
        pass

    def remove(self) -> ProcessState:
        pass
