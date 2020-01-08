
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from src.core.config_loader import Config
from src.sim.common.actors import ActorConfig
from src.sim.common.data_types import Action, State


@dataclass_json
@dataclass
class EnvironmentConfig(Config):
    actor_config: ActorConfig
    max_number_episode_steps: int = 100


class Environment:

    def __init__(self):
        pass

    def step(self, action: Action) -> State:
        pass

    def reset(self) -> State:
        pass
