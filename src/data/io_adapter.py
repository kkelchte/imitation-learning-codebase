
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from src.core.config_loader import Config
from src.sim.common.data_types import EnvironmentType


@dataclass_json
@dataclass
class IOAdapterConfig(Config):
    environment_type: EnvironmentType = None


class IOAdapter:
    pass
