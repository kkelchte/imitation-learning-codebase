"""Load dataset from local NFS file system and prepares it as pytorch dataset.

"""
import os
from dataclasses import dataclass
from typing import List, Generator, Optional

import numpy as np
from dataclasses_json import dataclass_json
from tqdm import tqdm

from src.core.config_loader import Config
from src.core.logger import cprint, get_logger, MessageType
from src.core.utils import get_filename_without_extension, get_data_dir
from src.core.data_types import Dataset
from src.data.utils import load_run, load_dataset_from_hdf5, balance_weights_over_actions, select
from src.core.data_types import Experience


@dataclass_json
@dataclass
class DataLoaderConfig(Config):
    data_directories: Optional[List[str]] = None
    hdf5_files: Optional[List[str]] = None
    random_seed: int = 123
    balance_over_actions: bool = False
    batch_size: int = 64
    subsample: int = 1
    loop_over_hdf5_files: bool = False
    input_size: Optional[List[int]] = None
    input_scope: Optional[str] = 'default'

    def post_init(self):  # add default options
        if self.data_directories is None:
            self.data_directories = []
        assert self.subsample >= 1
        if self.input_size is None:
            self.input_size = []
        if self.hdf5_files is None:
            self.hdf5_files = []

    def iterative_add_output_path(self, output_path: str) -> None:
        if self.output_path is None:
            if 'models' not in output_path:
                self.output_path = output_path
            else:  # if output path is provided by ModelConfig, the data should be found in the experiment directory
                self.output_path = output_path.split('models')[0]
        if self.data_directories is not None and len(self.data_directories) != 0 \
                and not self.data_directories[0].startswith('/'):
            self.data_directories = [os.path.join(get_data_dir(os.environ['HOME']), d) for d in self.data_directories]
        if self.hdf5_files is not None and len(self.hdf5_files) != 0:
            self.hdf5_files = [
                    os.path.join(get_data_dir(os.environ['HOME']), hdf5_f) if not hdf5_f.startswith('/') else hdf5_f
                    for hdf5_f in self.hdf5_files
            ]
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                value.iterative_add_output_path(output_path)


class DataLoader:

    def __init__(self, config: DataLoaderConfig):
        self._config = config
        self._logger = get_logger(name=get_filename_without_extension(__file__),
                                  output_path=config.output_path,
                                  quiet=False)
        cprint(f'Started.', self._logger)
        self._dataset = Dataset()
        self._num_runs = 0
        self._probabilities: List = []
        self.seed()
        self._hdf5_file_index = -1

    def seed(self, seed: int = None):
        np.random.seed(self._config.random_seed) if seed is None else np.random.seed(seed)

    def update_data_directories_with_raw_data(self):
        if self._config.data_directories is None:
            self._config.data_directories = []
        for d in sorted(os.listdir(os.path.join(self._config.output_path, 'raw_data'))):
            self._config.data_directories.append(os.path.join(self._config.output_path, 'raw_data', d))
        self._config.data_directories = list(set(self._config.data_directories))

    def load_dataset(self):
        if len(self._config.hdf5_files) != 0:
            if self._config.loop_over_hdf5_files:
                self._hdf5_file_index += 1
                self._hdf5_file_index %= len(self._config.hdf5_files)
                self._dataset = Dataset()
                self._dataset.extend(load_dataset_from_hdf5(self._config.hdf5_files[self._hdf5_file_index],
                                                            input_size=self._config.input_size))
                cprint(f'Loaded {len(self._dataset)} datapoints from {self._config.hdf5_files[self._hdf5_file_index]}',
                       self._logger,
                       msg_type=MessageType.warning if len(self._dataset.observations) == 0 else MessageType.info)
            else:
                for hdf5_file in self._config.hdf5_files:
                    self._dataset.extend(load_dataset_from_hdf5(hdf5_file,
                                                                input_size=self._config.input_size))
                cprint(f'Loaded {len(self._dataset)} datapoints from {self._config.hdf5_files}', self._logger,
                       msg_type=MessageType.warning if len(self._dataset.observations) == 0 else MessageType.info)
        else:
            self.load_dataset_from_directories(self._config.data_directories)

        if self._config.subsample != 1:
            self._dataset.subsample(self._config.subsample)

        if self._config.balance_over_actions:
            self._probabilities = balance_weights_over_actions(self._dataset)

    def load_dataset_from_directories(self, directories: List[str] = None) -> Dataset:
        directory_generator = tqdm(directories, ascii=True, desc=__name__) \
            if len(directories) > 10 else directories
        for directory in directory_generator:
            run = load_run(directory, arrange_according_to_timestamp=False, input_size=self._config.input_size,
                           scope=self._config.input_scope)
            if len(run) != 0:
                self._dataset.extend(experiences=run)
        cprint(f'Loaded {len(self._dataset)} data points from {len(directories)} directories',
               self._logger, msg_type=MessageType.warning if len(self._dataset) == 0 else MessageType.info)
        return self._dataset

    def empty_dataset(self) -> None:
        self._dataset = Dataset()

    def set_dataset(self, ds: Dataset = None) -> None:
        if ds is not None:
            self._dataset = ds
        else:
            self._dataset = Dataset()
            self.update_data_directories_with_raw_data()
            self.load_dataset()

    def get_dataset(self) -> Dataset:
        return self._dataset

    def get_data_batch(self) -> Generator[Dataset, None, None]:
        if len(self._dataset) == 0 or self._config.loop_over_hdf5_files:
            self.load_dataset()
        index = 0
        while index < len(self._dataset):
            batch = Dataset()
            end_index = min(index + self._config.batch_size, len(self._dataset)) \
                if self._config.batch_size != -1 else len(self._dataset)
            batch.observations = self._dataset.observations[index:end_index]
            batch.actions = self._dataset.actions[index:end_index]
            batch.done = self._dataset.done[index:end_index]
            batch.rewards = self._dataset.rewards[index:end_index]
            index = index + self._config.batch_size if self._config.batch_size != -1 else len(self._dataset)
            yield batch

    def sample_shuffled_batch(self, max_number_of_batches: int = 1000) \
            -> Generator[Dataset, None, None]:
        """
        randomly shuffle data samples in runs in dataset and provide them as ready run objects
        :param batch_size: number of samples or datapoints in one batch
        :param max_number_of_batches: define an upperbound in number of batches to end epoch
        :param dataset: list of runs with inputs, outputs and batches
        :return: yield a batch up until all samples are done
        """
        if len(self._dataset) == 0 or self._config.loop_over_hdf5_files:
            self.load_dataset()
        # Get data indices:
        batch_count = 0
        while batch_count < min(len(self._dataset), max_number_of_batches * self._config.batch_size):
            sample_indices = np.random.choice(list(range(len(self._dataset))),
                                              size=self._config.batch_size,
                                              p=self._probabilities
                                              if len(self._probabilities) != 0 else None)
            batch = select(self._dataset, sample_indices)
            batch_count += len(batch)
            yield batch
        return

    def split_data(self, indices: np.ndarray, *args) -> Generator[tuple, None, None]:
        """
        Split the indices in batches of configs batch_size and select the data in args.
        :param indices: possible indices to be selected. If all indices can be selected, provide empty array.
        :param args: lists or tensors from which the corresponding data according to the indices is selected.
        :return: provides a tuple in the same order as the args with the selected data.
        """
        if len(indices) == 0:
            indices = np.arange(len(self._dataset))
        np.random.shuffle(indices)
        splits = np.array_split(indices, max(1, int(len(self._dataset) / self._config.batch_size)))
        for selected_indices in splits:
            return_tuple = (select(data, selected_indices) for data in args)
            yield return_tuple

    def remove(self):
        [h.close() for h in self._logger.handlers]
