import os
import sys

from dataclasses import dataclass
from typing import Optional, List

from dataclasses_json import dataclass_json

from src.core.config_loader import Parser, Config
from src.core.data_types import Dataset
from src.data.data_loader import DataLoaderConfig, DataLoader
from src.data.utils import create_hdf5_file_from_dataset, set_binary_maps_as_target, augment_background_noise

"""
Data cleaner script loads raw data with a data loader, cleans it and stores the data in hdf5 files.
If the dataset exceeds 10G RAM approximately, the hdf5 files are stored in chunks (runs are never split).
"""


@dataclass_json
@dataclass
class DataCleaningConfig(Config):
    data_loader_config: DataLoaderConfig = None
    training_validation_split: float = 1.0
    remove_first_n_timestamps: Optional[int] = 0
    augment_background_noise: bool = False
    binary_maps_as_target: bool = False
    max_hdf5_size: int = 10**9

    def __post_init__(self):
        assert 0 <= self.training_validation_split <= 1


class DataCleaner:

    def __init__(self, config: DataCleaningConfig):
        self._config = config
        self._data_loader = DataLoader(config=config.data_loader_config)
        if len(config.data_loader_config.data_directories) == 0:
            self._data_loader.update_data_directories_with_raw_data()

    def clean(self):
        self._split_and_clean()

    def _split_and_clean(self):
        runs = self._config.data_loader_config.data_directories
        num_training_runs = int(len(runs) * self._config.training_validation_split)
        training_runs = self._config.data_loader_config.data_directories[:num_training_runs]
        validation_runs = self._config.data_loader_config.data_directories[num_training_runs:]
        for filename_tag, runs in zip(['train', 'validation'], [training_runs, validation_runs]):
            self._clean(filename_tag, runs)

    def _clean(self, filename_tag: str, runs: List[str]) -> None:
        filename_index = 0
        hdf5_data = Dataset()
        for run in runs:
            # load data in dataset in input size
            run_dataset = self._data_loader.load_dataset_from_directories([run])
            if len(run_dataset) <= self._config.remove_first_n_timestamps:
                continue
            # remove first N frames
            for _ in range(self._config.remove_first_n_timestamps):
                run_dataset.pop()
            # subsample
            run_dataset.subsample(self._config.data_loader_config.subsample)
            # augment with background noise and change target to binary map
            if self._config.binary_maps_as_target:
                run_dataset = set_binary_maps_as_target(run_dataset)
            if self._config.augment_background_noise:
                run_dataset = augment_background_noise(run_dataset)
            # store dhf5 file once max dataset size is reached
            hdf5_data.extend(run_dataset)
            self._data_loader.empty_dataset()
            if hdf5_data.get_memory_size() > self._config.max_hdf5_size:
                create_hdf5_file_from_dataset(filename=os.path.join(self._config.output_path,
                                                                    f'{filename_tag}_{filename_index}.hdf5'),
                                              dataset=hdf5_data)
                filename_index += 1
                hdf5_data = Dataset()
        if len(hdf5_data) != 0:
            create_hdf5_file_from_dataset(filename=os.path.join(self._config.output_path,
                                                                f'{filename_tag}_{filename_index}.hdf5'),
                                          dataset=hdf5_data)


if __name__ == '__main__':
    config_file = Parser().parse_args().config
    data_cleaning_config = DataCleaningConfig().create(config_file=config_file)
    data_cleaner = DataCleaner(config=data_cleaning_config)
    data_cleaner.clean()
