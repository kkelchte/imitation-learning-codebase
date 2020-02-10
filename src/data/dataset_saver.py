import os
from typing import List

from datetime import datetime
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from src.core.config_loader import Config
from src.core.logger import get_logger, cprint
from src.data.utils import timestamp_to_filename, store_image, store_array_to_file
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
    saving_directory: str = None
    sensors: List[str] = None
    actors: List[str] = None

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
            self.saving_directory = os.path.join(self.output_path, 'raw_data',
                                                 f'{datetime.strftime(datetime.now(), format="%y-%m-%d_%H-%M-%S")}')


class DataSaver:

    def __init__(self, config: DataSaverConfig):
        self._config = config
        self._logger = get_logger(name=__name__,
                                  output_path=self._config.output_path,
                                  quite=False)
        cprint(f'initiate', self._logger)

        if not self._config.saving_directory.startswith('/'):
            self._config.saving_directory = os.path.join(os.environ['HOME'],
                                                         self._config.saving_directory)

    def update_saving_directory(self):
        self._config.saving_directory = os.path.join(os.path.dirname(self._config.saving_directory),
                                                     datetime.strftime(datetime.now(), format="%y-%m-%d_%H-%M-%S"))

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
