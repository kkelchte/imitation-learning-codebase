import torch
from dataclasses import dataclass
import numpy as np
import yaml
from dataclasses_json import dataclass_json

from src.ai.model import Model, ModelConfig
from src.core.config_loader import Config
from src.core.logger import get_logger
from src.core.utils import get_filename_without_extension
from src.sim.common.data_types import Action, ActorType


"""Base actor class and dnn actor class

Depends on ai/architectures/models to call forward pass.
"""


@dataclass_json
@dataclass
class ActorConfig(Config):
    name: str = ''
    type: ActorType = ActorType.Model
    model_config: ModelConfig = None
    specs: dict = None
    file: str = None

    def __post_init__(self):
        if self.file is not None:
            with open(self.file, 'r') as f:
                specs = yaml.load(f, Loader=yaml.FullLoader)
                self.specs = specs['specs'] if 'specs' in specs.keys() else specs
            del self.model_config  # loaded from file into specs
        else:
            del self.specs
            del self.file


class Actor:

    def __init__(self, config: ActorConfig):
        self._config = config
        self._name = config.name
        self._type = config.type

    def get_action(self, sensor_data: dict) -> Action:
        pass

    def get_name(self):
        return self._name


class DNNActor(Actor):

    def __init__(self, config: ActorConfig):
        super().__init__(config)
        self._logger = get_logger(get_filename_without_extension(__file__), self._config.output_path)
        self._model = Model(config=self._config.model_config)

    def get_action(self, sensor_data: dict) -> Action:
        processed_image = torch.Tensor(sensor_data['observation']).permute(2, 0, 1).unsqueeze(0)
        assert processed_image.size()[0] == 1 and processed_image.size()[1] == 3
        output = self._model.forward([processed_image], train=False)[0].detach().cpu().numpy()
        return Action(
            actor_name='dnn_actor',
            actor_type=ActorType.Model,
            value=output  # assuming control is first output
        )
