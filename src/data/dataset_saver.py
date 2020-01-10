import os
from typing import List

from dataclasses import dataclass
from dataclasses_json import dataclass_json
from PIL import Image

from src.core.config_loader import Config
from src.core.logger import get_logger
from src.data.io_adapter import convert_state_to_frame, timestamp_to_filename
from src.data.data_types import Frame, SensorType
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
    sensors: List[SensorType] = None


class DataSaver:

    def __init__(self, config: DataSaverConfig):
        self._config = config
        logger = get_logger(name=__name__,
                            output_path=self._config.output_path,
                            quite=False)
        logger.info(f'initiate')
        if not self._config.saving_directory.startswith('/'):
            self._config.saving_directory = os.path.join(self._config.output_path,
                                                         self._config.saving_directory)
        if not os.path.isdir(self._config.saving_directory):
            os.makedirs(self._config.saving_directory)
        for sensor in self._config.sensors:
            os.makedirs(os.path.join(self._config.saving_directory,
                                     sensor.name), exist_ok=True)

    def save(self, state: State, action: Action = None) -> None:
        for sensor in state.sensor_data.keys():
            if sensor in self._config.sensors:
                frame = convert_state_to_frame(raw_data=state.sensor_data[sensor],
                                               sensor_type=sensor,
                                               time_stamp_us=state.time_stamp_us)
                self.store_frame(frame)
        # self.store_action(action)

    # def store_action(self, action: Action):
    #     file_name =
    #     with open(file_name)

    def store_frame(self, frame: Frame) -> None:
        store_factory = {
            'rgb': self.store_rgb,
        }
        return store_factory[frame.sensor_type](frame)

    def store_rgb(self, frame):
        file_name = os.path.join(self._config.saving_directory, frame.sensor,
                                 timestamp_to_filename(frame.time_stamp_us) + '.jpg')

        im = Image.fromarray(frame.data)
        im.save(file_name)
