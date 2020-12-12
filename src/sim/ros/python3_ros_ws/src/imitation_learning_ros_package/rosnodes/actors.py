from typing import Optional

from dataclasses import dataclass
import yaml
from dataclasses_json import dataclass_json

from src.core.config_loader import Config
from src.core.data_types import Action


"""Base actor class and dnn actor class

Depends on ai/architectures/models to call forward pass.
"""


@dataclass_json
@dataclass
class ActorConfig(Config):
    name: str = ''
    specs: Optional[dict] = None
    file: Optional[str] = None

    def __post_init__(self):
        """Actor either contains a file from which specifications are loaded, used by ROS actors
        or it contains a model config. Delete the unused None variables."""
        if self.file is not None:
            with open(self.file, 'r') as f:
                specs = yaml.load(f, Loader=yaml.FullLoader)
                self.specs = specs['specs'] if 'specs' in specs.keys() else specs
        else:
            del self.file
        if self.specs is None:
            del self.specs


class Actor:

    def __init__(self, config: ActorConfig):
        self._config = config
        self._name = config.name if config.name != '' else 'no_name'

    def get_name(self):
        return self._name

    def run(self):
        pass
