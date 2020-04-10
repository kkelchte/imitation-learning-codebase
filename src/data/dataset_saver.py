import os
import time
from typing import List

from datetime import datetime
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from src.core.config_loader import Config
from src.core.logger import get_logger, cprint, MessageType
from src.core.utils import get_date_time_tag, get_filename_without_extension
from src.data.utils import timestamp_to_filename, store_image, store_array_to_file, create_hdf5_file
from src.data.data_types import Frame
from src.sim.common.data_types import State, Action, TerminalType

"""Stores experiences as episodes in dataset.

Package shared by real-world and simulated experiments.
Stores experiences with compressed states.
Extension:
    - convert finished dataset to hdf5 file
"""


@dataclass_json
@dataclass
class DataSaverConfig(Config):
    saving_directory_tag: str = ''
    saving_directory: str = None
    sensors: List[str] = None
    actors: List[str] = None
    training_validation_split: float = 0.9
    store_hdf5: bool = False

    def __post_init__(self):
        if self.sensors is None:
            self.sensors = ['all']
        if self.actors is None:
            self.actors = ['all']

    def iterative_add_output_path(self, output_path: str) -> None:
        self.output_path = output_path
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                value.iterative_add_output_path(output_path)
        if self.saving_directory is None:
            self.saving_directory = create_saving_directory(self.output_path, self.saving_directory_tag)


def create_saving_directory(output_path: str, saving_directory_tag: str = ''):
    saving_directory = os.path.join(output_path, 'raw_data', f'{get_date_time_tag()}')
    if saving_directory_tag != '':
        saving_directory += f'_{saving_directory_tag}'
    while os.path.isdir(saving_directory):  # update second if directory is already taken
        time.sleep(1)
        saving_directory = os.path.join(output_path, 'raw_data', f'{get_date_time_tag()}')
        if saving_directory_tag != '':
            saving_directory += f'_{saving_directory_tag}'
    os.makedirs(saving_directory)
    return saving_directory


class DataSaver:

    def __init__(self, config: DataSaverConfig):
        self._config = config
        self._logger = get_logger(name=get_filename_without_extension(__file__),
                                  output_path=self._config.output_path,
                                  quite=False)
        cprint(f'initiate', self._logger)

        if not self._config.saving_directory.startswith('/'):
            self._config.saving_directory = os.path.join(os.environ['HOME'],
                                                         self._config.saving_directory)

    def update_saving_directory(self):
        self._config.saving_directory = create_saving_directory(self._config.output_path,
                                                                self._config.saving_directory_tag)

    def get_saving_directory(self):
        return self._config.saving_directory

    def save(self, state: State, action: Action = None) -> None:
        if state.terminal == TerminalType.Unknown:
            return
        for sensor in state.sensor_data.keys():
            if sensor in self._config.sensors or self._config.sensors == ['all']:
                # TODO make multitasked with asyncio
                self._store_frame(
                    Frame(origin=sensor,
                          time_stamp_ms=state.time_stamp_ms,
                          data=state.sensor_data[sensor]
                          )
                )
        for actor in state.actor_data.keys():
            if actor in self._config.actors or self._config.actors == ['all']:
                if state.actor_data[actor].value is not None:
                    self._store_frame(
                        Frame(origin=actor,
                              time_stamp_ms=state.time_stamp_ms,
                              data=state.actor_data[actor].value
                              )
                    )
        if action is not None:
            self._store_frame(
                Frame(
                    origin='action',
                    time_stamp_ms=state.time_stamp_ms,
                    data=action.value
                )
            )
        if state.terminal in [TerminalType.Success, TerminalType.Failure]:
            self._store_termination(state.terminal)

    def _store_termination(self, terminal: TerminalType, time_stamp: int = -1) -> None:
        msg = f'{time_stamp}: ' if time_stamp != -1 else ''
        msg += terminal.name + '\n'
        with open(os.path.join(self._config.saving_directory, 'termination'), 'w') as f:
            f.write(msg)

    def _store_frame(self, frame: Frame) -> None:
        if len(frame.data.shape) in [2, 3]:
            if not os.path.isdir(os.path.join(self._config.saving_directory, frame.origin)):
                os.makedirs(os.path.join(self._config.saving_directory, frame.origin), exist_ok=True)
            store_image(data=frame.data, file_name=os.path.join(self._config.saving_directory, frame.origin,
                                                                timestamp_to_filename(frame.time_stamp_ms)) + '.jpg')
        elif len(frame.data.shape) == 1:
            store_array_to_file(data=frame.data, file_name=os.path.join(self._config.saving_directory, frame.origin),
                                time_stamp=frame.time_stamp_ms)

    def create_train_validation_hdf5_files(self) -> None:
        if not self._config.store_hdf5:
            cprint(f'store_hdf5: {self._config.store_hdf5}', self._logger, msg_type=MessageType.warning)
            return
        raw_data_dir = os.path.dirname(self._config.saving_directory)
        runs = [
            os.path.join(raw_data_dir, run)
            for run in sorted(os.listdir(raw_data_dir))
        ]
        number_of_training_runs = int(self._config.training_validation_split*len(runs))
        train_runs = runs[0:number_of_training_runs]
        validation_runs = runs[number_of_training_runs:]
        create_hdf5_file(filename=os.path.join(self._config.output_path, 'train.hdf5'), runs=train_runs)
        create_hdf5_file(filename=os.path.join(self._config.output_path, 'validation.hdf5'), runs=validation_runs)
