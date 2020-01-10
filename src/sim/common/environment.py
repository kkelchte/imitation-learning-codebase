
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from src.core.config_loader import Config
from src.sim.common.data_types import Action, State, EnvironmentType
from src.sim.common.actors import Actor


@dataclass_json
@dataclass
class EnvironmentConfig(Config):
    name: str = None
    environment_type: EnvironmentType = None
    max_number_of_steps: int = 100


class Environment:

    def __init__(self):
        pass

    def step(self, action: Action) -> State:
        pass

    def reset(self) -> State:
        pass

    def get_actor(self) -> Actor:
        pass
