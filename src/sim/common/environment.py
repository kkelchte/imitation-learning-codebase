
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from src.core.config_loader import Config
from src.sim.common.data_types import Action, State, EnvironmentType
from src.sim.common.actors import Actor


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
    world_name: str = None
    robot_name: str = None
    turtlebot_sim: bool = False


@dataclass_json
@dataclass
class RosConfig(Config):
    """
    Configuration specific for ROS environment,
    specified here to avoid circular dependencies environment <> ros_environment
    """
    headless: bool = False
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

    def step(self, action: Action) -> State:
        pass

    def reset(self) -> State:
        pass

    def get_actor(self) -> Actor:
        pass
