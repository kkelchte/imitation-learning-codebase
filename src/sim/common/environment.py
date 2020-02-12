import os
from dataclasses import dataclass
from typing import List, Union

from dataclasses_json import dataclass_json

from src.core.config_loader import Config
from src.core.logger import get_logger, cprint
from src.sim.common.data_types import Action, State, EnvironmentType, ActorType, ProcessState
from src.sim.common.actors import Actor, ActorConfig


@dataclass_json
@dataclass
class GymConfig(Config):
    """
    Configuration specific for Gym environment,
    specified here to avoid circular dependencies environment <> gym_environment
    """
    random_seed: int = 123
    world_name: str = None


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
    step_rate_fps: float = 10.
    visible_xterm: bool = False
    ros_launch_config: RosLaunchConfig = None


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
    actor_configs: List[ActorConfig] = None
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
        self._logger = get_logger(name=__name__,
                                  output_path=self._config.output_path,
                                  quite=False)
        cprint(f'initiate', self._logger)

    def step(self, action: Action) -> State:
        pass

    def reset(self) -> State:
        pass

    def get_actor(self) -> Union[Actor, ActorType]:
        pass

    def remove(self) -> ProcessState:
        pass
