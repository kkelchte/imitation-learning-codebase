from dataclasses import dataclass
import numpy as np
import yaml

from src.sim.common.data_types import Action, ActorType


"""Base actor class and dnn actor class

Depends on ai/architectures/models to call forward pass.
"""


@dataclass
class ActorConfig:
    name: str = None
    type: ActorType = None
    specs: dict = None
    file: str = None

    def __post_init__(self):
        if self.specs is None and self.file is not None:
            with open(self.file, 'r') as f:
                specs = yaml.load(f, Loader=yaml.FullLoader)
                self.specs = specs['specs'] if 'specs' in specs.keys() else specs


class Actor:

    def __init__(self, config: ActorConfig):
        self._name = config.name
        self._type = config.type
        self._specs = config.specs
        self._config_file = config.file

    def get_action(self, sensor_data: dict = None) -> Action:
        pass

    def get_name(self):
        return self._name


# class DNNActor(Actor):
#
#     def __init__(self, config: ActorConfig):
#         super().__init__(config)
#         self._dnn = BaseModel.load_from_trace_path(self._config.actor_specs['model_trace_path'])
#
#     def get_action(self, sensor_data: dict) -> Action:
#         processed_state = self.io_adapter.from_raw_to_(sensor_data)
#         return Action(self._dnn.forward(processed_state))
