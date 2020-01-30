"""Load dataset from local NFS file system and prepares it as pytorch dataset.

"""
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List

from dataclasses_json import dataclass_json

from src.core.config_loader import Config


@dataclass_json
@dataclass
class DataLoaderConfig(Config):
    saving_directory: str = None
    sensors: List[str] = None
    actors: List[str] = None

    def __post_init__(self):
        if self.sensors is None:
            self.sensors = ['all']
        if self.actors is None:
            self.actors = ['all']
        if self.saving_directory is None and self.output_path is not None:
            self.saving_directory = os.path.join(self.output_path, 'raw_data',
                                                 datetime.strftime(datetime.now(), format="%y-%m-%d_%H-%M-%S"))


class DataLoader:

    def __init__(self, config: DataLoaderConfig):
        pass
