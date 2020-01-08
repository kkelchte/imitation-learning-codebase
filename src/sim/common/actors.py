from dataclasses import dataclass
import numpy as np

from src.sim.common.data_types import Action
from src.data.io_adapter import IOAdapterConfig, IOAdapter


"""Base actor class and dnn actor class

Depends on ai/architectures/models to call forward pass.
"""


@dataclass
class ActorConfig:
    name: str


class Actor:

    def __init__(self, config: ActorConfig):
        self._name = config.name

    def get_action(self, raw_state: np.array, visualize: bool = False, verbose: bool = False) -> Action:
        pass

    def get_name(self):
        return self._name


@dataclass
class DNNActorConfig(ActorConfig):
    model_trace_path: str
    io_adapter_config: IOAdapterConfig

class DNNActor(Actor):

    def __init__(self):
        super(DNNActor, self).__init__()
        #self._dnn = torch.load(model_trace_path)
        self.io_adapter = IOAdapter(config.io_adapter_config)

    def get_action(self, raw_state: np.array, visualize: bool = False, verbose: bool = False) -> Action:
        processed_state = self.io_adapter(raw_state)
        return Action(self._dnn.forward(processed_state))
