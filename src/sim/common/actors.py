from dataclasses import dataclass
import numpy as np

from src.sim.common.data_types import Action, ActorType
from src.ai.architectures.models.model import BaseModel


"""Base actor class and dnn actor class

Depends on ai/architectures/models to call forward pass.
"""


@dataclass
class ActorConfig:
    description: str = None
    actor_type: ActorType = None


class Actor:

    def __init__(self, config: ActorConfig):
        self._description = config.description

    def get_action(self, raw_state: np.array, visualize: bool = False, verbose: bool = False) -> Action:
        pass

    def get_description(self):
        return self._description


@dataclass
class DNNActorConfig(ActorConfig):
    model_trace_path: str = None


class DNNActor(Actor):

    def __init__(self, config: DNNActorConfig):
        super(DNNActor, self).__init__(config)
        self._config = config
        self._dnn = BaseModel.load_from_trace_path(self._config.model_trace_path)

    def get_action(self, raw_state: np.array, visualize: bool = False, verbose: bool = False) -> Action:
        processed_state = self.io_adapter.from_raw_to_(raw_state)
        return Action(self._dnn.forward(processed_state))
