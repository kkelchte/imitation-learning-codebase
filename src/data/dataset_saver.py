import os

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from src.core.config_loader import Config
from src.core.logger import get_logger
from src.sim.common.data_types import State, Action

"""Stores experiences as episodes in dataset.

Package shared by real-world and simulated experiments.
Stores experiences with compressed states.
Extension:
    - convert finished dataset to hdf5 file
"""


@dataclass_json
@dataclass
class DataSaverConfig(Config):
    saving_directory: str = None
    store_RGB: bool = False
    store_depth: bool = False
    store_pose: bool = False


class DataSaver:

    def __init__(self, config: DataSaverConfig):
        logger = get_logger(name=__name__,
                            output_path=config.output_path,
                            quite=False)
        logger.info(f'initiate')
        if not os.path.isdir(config.output_path):
            os.makedirs(config.output_path)

    def save(self, state: State, action: Action = None):
        # TODO save state and action as data frames
        pass
