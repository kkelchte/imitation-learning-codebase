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
from src.core.utils import get_filename_without_extension
from src.core.data_types import Dataset
from src.data.utils import load_run, load_dataset_from_hdf5, balance_weights_over_actions, select
from src.core.data_types import Experience


@dataclass_json
@dataclass
class DataLoaderConfig(Config):
    data_directories: Optional[List[str]] = None
    hdf5_file: str = ''
    random_seed: int = 123
    balance_over_actions: bool = False
    batch_size: int = 64
    subsample: int = 1
    input_size: Optional[List[int]] = None

    def post_init(self):  # add default options
        if self.data_directories is None:
            self.data_directories = []
        assert self.subsample >= 1

    def iterative_add_output_path(self, output_path: str) -> None:
        if self.output_path is None:
            if 'models' not in output_path:
                self.output_path = output_path
            else:  # if output path is provided by ModelConfig, the data should be found in the experiment directory
                self.output_path = output_path.split('models')[0]
        if self.data_directories is not None and len(self.data_directories) != 0 \
                and not self.data_directories[0].startswith('/'):
            self.data_directories = [os.path.join(self.output_path, d) for d in self.data_directories]
        if len(self.hdf5_file) != 0 and not self.hdf5_file.startswith('/'):
            self.hdf5_file = os.path.join(self.output_path, self.hdf5_file)
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

    def seed(self, seed: int = None):
        np.random.seed(self._config.random_seed) if seed is None else np.random.seed(seed)

    def update_data_directories_with_raw_data(self):
        if self._config.data_directories is None:
            self._config.data_directories = []
        for d in sorted(os.listdir(os.path.join(self._config.output_path, 'raw_data'))):
            self._config.data_directories.append(os.path.join(self._config.output_path, 'raw_data', d))
        self._config.data_directories = list(set(self._config.data_directories))

    def load_dataset(self, arrange_according_to_timestamp: bool = False):
        if self._config.hdf5_file != '':
            self._dataset = load_dataset_from_hdf5(self._config.hdf5_file, size=self._config.input_size)
            cprint(f'Loaded {len(self._dataset.observations)} from {self._config.hdf5_file}', self._logger,
                   msg_type=MessageType.warning if len(self._dataset.observations) == 0 else MessageType.info)
        else:
            directory_generator = tqdm(self._config.data_directories, ascii=True, desc=__name__) \
                if len(self._config.data_directories) > 10 else self._config.data_directories
            for directory in directory_generator:
                run = load_run(directory, arrange_according_to_timestamp, input_size=self._config.input_size)
                if len(run) != 0:
                    self._dataset.extend(experiences=run)
            cprint(f'Loaded {len(self._dataset)} data points from {len(self._config.data_directories)} directories',
                   self._logger, msg_type=MessageType.warning if len(self._dataset) == 0 else MessageType.info)

        if self._config.subsample != 1:
            self._dataset.subsample(self._config.subsample)

        if self._config.balance_over_actions:
            self._probabilities = balance_weights_over_actions(self._dataset)

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
        index = 0
        while index < len(self._dataset):
            batch = Dataset()
            end_index = min(index + self._config.batch_size, len(self._dataset)) \
                if self._config.batch_size != -1 else len(self._dataset)
            batch.observations = self._dataset.observations[index:end_index]
            batch.actions = self._dataset.actions[index:end_index]
            batch.done = self._dataset.done[index:end_index]
            batch.rewards = self._dataset.rewards[index:end_index]
            index += self._config.batch_size
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
        if len(self._dataset) == 0:
            msg = f'Cannot sample batch from dataset of size {len(self._dataset)}, ' \
                  f'make sure you call DataLoader.load_dataset()'
            cprint(msg, self._logger, msg_type=MessageType.error)
            raise IOError(msg)
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
