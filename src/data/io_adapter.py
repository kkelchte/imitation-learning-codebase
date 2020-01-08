
from dataclasses import dataclass

from src.sim.common.data_types import EnvironmentType


@dataclass
class IOAdapterConfig:
    environment_type: EnvironmentType
