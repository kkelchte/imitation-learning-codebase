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
    actor_specs: dict = None


class Actor:

    def __init__(self, config: ActorConfig):
        self._config = config
        self._description = config.description

    def get_action(self, sensor_data: dict) -> Action:
        pass

    def get_description(self):
        return self._description


class DNNActor(Actor):

    def __init__(self, config: ActorConfig):
        super().__init__(config)
        self._dnn = BaseModel.load_from_trace_path(self._config.actor_specs['model_trace_path'])

    def get_action(self, sensor_data: dict) -> Action:
        processed_state = self.io_adapter.from_raw_to_(sensor_data)
        return Action(self._dnn.forward(processed_state))
