import os
import sys
import random

from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, List

from dataclasses_json import dataclass_json

from src.core.config_loader import Parser, Config
from src.core.data_types import Dataset
from src.data.data_loader import DataLoaderConfig, DataLoader
from src.data.utils import create_hdf5_file_from_dataset, set_binary_maps_as_target, augment_background_noise, \
    augment_background_textured, parse_binary_maps

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
    augment_background_noise: float = 0.
    augment_background_textured: float = 0.
    augment_empty_images: float = 0.
    texture_directory: str = ""  # directory in to fill background with
    binary_maps_as_target: bool = False  # extract binary maps from inputs and store in action field in hdf5
    smoothen_labels: bool = False
    # binary maps are extracted according to a threshold, in line_world bg is white (high), line is blue (low)
    # so data is best inverted to predict line high and bg low
    invert_binary_maps: bool = False
    require_success: bool = False  # skip runs without a SUCCESS tag in their raw data run directory
    shuffle: bool = False  # shuffle dataset before exporting it as hdf5, used for validation images
    max_hdf5_size: int = 10**9
    max_run_length: int = -1

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
        shuffled_runs = self._config.data_loader_config.data_directories[:]
        random.shuffle(shuffled_runs)
        num_training_runs = int(len(shuffled_runs) * self._config.training_validation_split)
        training_runs = shuffled_runs[:num_training_runs]
        validation_runs = shuffled_runs[num_training_runs:]
        for filename_tag, runs in zip(['train', 'validation'], [training_runs, validation_runs]):
            self._clean(filename_tag, runs)

    def _clean(self, filename_tag: str, runs: List[str]) -> None:
        total_data_points = 0
        filename_index = 0
        hdf5_data = Dataset()
        for run in tqdm(runs):
            if self._config.require_success:
                if not os.path.isfile(os.path.join(run, 'Success')):
                    continue
            # load data in dataset in input size
            run_dataset = self._data_loader.load_dataset_from_directories([run])
            if len(run_dataset) <= self._config.remove_first_n_timestamps:
                continue
            # remove first N frames
            for _ in range(self._config.remove_first_n_timestamps):
                run_dataset.pop()
            # subsample
            run_dataset.subsample(self._config.data_loader_config.subsample)
            # enforce max run length
            if self._config.max_run_length != -1:
                run_dataset.clip(self._config.max_run_length)
                assert len(run_dataset) <= self._config.max_run_length
            # augment with background noise and change target to binary map

            binary_maps = parse_binary_maps(run_dataset.observations, invert=self._config.invert_binary_maps) \
                if self._config.augment_background_noise != 0 or self._config.augment_background_textured != 0 else None
            if self._config.binary_maps_as_target:
                run_dataset = set_binary_maps_as_target(run_dataset, invert=self._config.invert_binary_maps,
                                                        binary_images=binary_maps,
                                                        smoothen_labels=self._config.smoothen_labels)

            if self._config.augment_background_noise != 0:
                run_dataset = augment_background_noise(run_dataset, p=self._config.augment_background_noise,
                                                       binary_images=binary_maps)
            if self._config.augment_background_textured != 0:
                run_dataset = augment_background_textured(run_dataset,
                                                          texture_directory=self._config.texture_directory,
                                                          p=self._config.augment_background_textured,
                                                          p_empty=self._config.augment_empty_images,
                                                          binary_images=binary_maps)
            # store dhf5 file once max dataset size is reached
            hdf5_data.extend(run_dataset)
            self._data_loader.empty_dataset()
            if hdf5_data.get_memory_size() > self._config.max_hdf5_size:
                if self._config.shuffle:
                    hdf5_data.shuffle()
                create_hdf5_file_from_dataset(filename=os.path.join(self._config.output_path,
                                                                    f'{filename_tag}_{filename_index}.hdf5'),
                                              dataset=hdf5_data)
                filename_index += 1
                total_data_points += len(hdf5_data)
                hdf5_data = Dataset()
        if len(hdf5_data) != 0:
            if self._config.shuffle:
                hdf5_data.shuffle()
            create_hdf5_file_from_dataset(filename=os.path.join(self._config.output_path,
                                                                f'{filename_tag}_{filename_index}.hdf5'),
                                          dataset=hdf5_data)
            total_data_points += len(hdf5_data)
        print(f'Total data points: {total_data_points}')


if __name__ == '__main__':
    config_file = Parser().parse_args().config
    data_cleaning_config = DataCleaningConfig().create(config_file=config_file)
    data_cleaner = DataCleaner(config=data_cleaning_config)
    data_cleaner.clean()
