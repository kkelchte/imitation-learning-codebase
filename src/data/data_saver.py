import os
import shutil
from typing import Optional

import numpy as np
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from src.core.config_loader import Config
from src.core.logger import get_logger, cprint, MessageType
from src.core.utils import get_date_time_tag, get_filename_without_extension
from src.data.data_loader import DataLoaderConfig, DataLoader
from src.data.utils import timestamp_to_filename, store_image, store_array_to_file, create_hdf5_file_from_dataset
from src.core.data_types import Dataset, Action
from src.core.data_types import Experience, TerminationType

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
    saving_directory: Optional[str] = None
    training_validation_split: float = 0.9
    max_size: int = -1
    store_hdf5: bool = False
    store_on_ram_only: bool = False
    clear_buffer_before_episode: bool = False
    separate_raw_data_runs: bool = False

    def __post_init__(self):
        assert not (self.store_hdf5 and self.store_on_ram_only)

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
    original_saving_directory = saving_directory
    count = 0
    while os.path.isdir(saving_directory):  # add _0 _1 if directory already exists.
        saving_directory = original_saving_directory + '_' + str(count)
        count += 1
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

        if self._config.store_on_ram_only:
            self._dataset = Dataset(max_size=self._config.max_size)

        self._frame_counter = 0  # used to keep track of replay buffer size on file system
        #  TODO: count frames that are already in raw_data

    def __len__(self):
        if self._config.store_on_ram_only:
            return len(self._dataset)
        else:
            return self._frame_counter

    def update_saving_directory(self):
        self._config.saving_directory = create_saving_directory(self._config.output_path,
                                                                self._config.saving_directory_tag)

    def get_saving_directory(self):
        return self._config.saving_directory if not self._config.store_on_ram_only else 'ram'

    def get_dataset(self):
        return self._dataset

    def save(self, experience: Experience) -> None:
        if experience.done == TerminationType.Unknown:
            return  # don't save experiences in an unknown state
        if self._config.store_on_ram_only:
            return self._dataset.append(experience)
        else:
            return self._store_in_file_system(experience=experience)

    def _store_in_file_system(self, experience: Experience) -> None:
        for dst, data in zip(['observation', 'action', 'reward', 'done'],
                             [experience.observation, experience.action, experience.reward, experience.done]):
            self._store_frame(data=np.asarray(data.value if isinstance(data, Action) else data),
                              dst=dst, time_stamp=experience.time_stamp)

        for key, value in experience.info.items():
            self._store_frame(data=np.asarray(value), dst=key, time_stamp=experience.time_stamp)

        if experience.done in [TerminationType.Success, TerminationType.Failure]:
            os.system(f'touch {os.path.join(self._config.saving_directory, experience.done.name)}')
        self._check_dataset_size_on_file_system()

    def _store_frame(self, data: np.ndarray, dst: str, time_stamp: int) -> None:
        if len(data.shape) in [2, 3]:
            if not os.path.isdir(os.path.join(self._config.saving_directory, dst)):
                os.makedirs(os.path.join(self._config.saving_directory, dst), exist_ok=True)
            store_image(data=data, file_name=os.path.join(self._config.saving_directory, dst,
                                                          timestamp_to_filename(time_stamp)) + '.jpg')
        elif len(data.shape) in [0, 1]:
            store_array_to_file(data=data, file_name=os.path.join(self._config.saving_directory, dst + '.data'),
                                time_stamp=time_stamp)

    def _check_dataset_size_on_file_system(self):
        self._frame_counter += 1
        # If number of frames exceed max_size, remove oldest run and decrease frame counter
        if self._frame_counter > self._config.max_size != -1:
            raw_data_dir = os.path.dirname(self._config.saving_directory)
            first_run = sorted(os.listdir(raw_data_dir))[0]
            with open(os.path.join(raw_data_dir, first_run, 'done.data'), 'r') as f:
                run_length = len(f.readlines())
            self._frame_counter -= run_length
            shutil.rmtree(os.path.join(raw_data_dir, first_run), ignore_errors=True)
            if not self._config.separate_raw_data_runs:
                cprint(f"Reached max buffer size and removing all data."
                       f"Avoid this by setting data_saver_config.separate_raw_data_runs to True.",
                       msg_type=MessageType.warning, logger=self._logger)

    def create_train_validation_hdf5_files(self) -> None:
        raw_data_dir = os.path.dirname(self._config.saving_directory)
        all_runs = [
            os.path.join(raw_data_dir, run)
            for run in sorted(os.listdir(raw_data_dir))
        ]
        number_of_training_runs = int(self._config.training_validation_split*len(all_runs))
        train_runs = all_runs[0:number_of_training_runs]
        validation_runs = all_runs[number_of_training_runs:]

        for file_name, runs in zip(['train', 'validation'], [train_runs, validation_runs]):
            config = DataLoaderConfig().create(config_dict={
                'data_directories': runs,
                'output_path': self._config.output_path,
            })
            data_loader = DataLoader(config=config)
            data_loader.load_dataset(arrange_according_to_timestamp=False)
            create_hdf5_file_from_dataset(filename=os.path.join(self._config.output_path, file_name + '.hdf5'),
                                          dataset=data_loader.get_dataset())

    def empty_raw_data_in_output_directory(self) -> None:
        raw_data_directory = os.path.dirname(self._config.saving_directory)
        for d in os.listdir(raw_data_directory):
            shutil.rmtree(os.path.join(raw_data_directory, d))

    def clear_buffer(self) -> None:
        self._frame_counter = 0
        if self._config.store_on_ram_only:
            self._dataset = Dataset()
        else:
            self.empty_raw_data_in_output_directory()

    def remove(self):
        [h.close() for h in self._logger.handlers]
