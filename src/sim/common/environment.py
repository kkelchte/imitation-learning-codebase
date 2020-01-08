
from dataclasses import dataclass

from src.sim.common.actors import ActorConfig
from src.sim.common.data_types import Action, State


@dataclass
class EnvironmentConfig:
    actor_config: ActorConfig
    max_number_episode_steps: int = 100


class Environment:

    def __init__(self):
        pass

    def step(self, action: Action) -> State:
        pass

    def reset(self) -> State:
        pass
