
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from src.core.config_loader import Config
from src.sim.common.data_types import Action, State, EnvironmentType
from src.sim.common.actors import Actor


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
    robot_name: str = 'turtle'
    world_name: str = 'esat'


class Environment:

    def __init__(self, config: EnvironmentConfig):
        self._config = config

    def step(self, action: Action) -> State:
        pass

    def reset(self) -> State:
        pass

    def get_actor(self) -> Actor:
        pass
