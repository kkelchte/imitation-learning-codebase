"""Load dataset from local NFS file system and prepares it as pytorch dataset.

"""
import os
from dataclasses import dataclass
from typing import List, Tuple, Generator

import numpy as np
import torch
from dataclasses_json import dataclass_json
from tqdm import tqdm

from src.core.config_loader import Config
from src.core.logger import cprint, get_logger, MessageType
from src.core.utils import get_filename_without_extension
from src.data.data_types import Dataset, Run
from src.data.utils import load_and_preprocess_file, filename_to_timestamp, torch_append, \
    arrange_run_according_timestamps, load_data, load_dataset_from_hdf5, calculate_probabilites_per_run


@dataclass_json
@dataclass
class DataLoaderConfig(Config):
    data_directories: List[str] = None
    hdf5_file: str = ''
    inputs: List[str] = None
    outputs: List[str] = None
    reward: str = ''
    balance_targets: bool = False
    data_sampling_seed: int = 123

    def post_init(self):  # add default options
        if self.inputs is None:
            self.inputs = ['forward_camera']
        if self.outputs is None:
            self.outputs = ['ros_expert']
        if self.data_directories is None:
            del self.data_directories

    def iterative_add_output_path(self, output_path: str) -> None:
        if self.output_path is None:
            self.output_path = output_path
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
                                  quite=False)
        cprint(f'Started.', self._logger)
        np.random.seed(self._config.data_sampling_seed)
        self._dataset = Dataset()
        self._num_datapoints = 0
        self._num_runs = 0
        self._probabilities_per_run: List[List] = []  # sample probabilities per data point in a run
        self._run_probabilities: List = []  # normalize over different runs according to number of samples in run

    def load_dataset(self, input_sizes: List[List] = None, output_sizes: List[List] = None):
        if self._config.hdf5_file is not '':
            self._dataset = load_dataset_from_hdf5(self._config.hdf5_file,
                                                   self._config.inputs,
                                                   self._config.outputs)
            cprint(f'Loaded {len(self._dataset.data)} from {self._config.hdf5_file}',
                   self._logger, msg_type=MessageType.error if len(self._dataset.data) == 0 else MessageType.info)
        else:
            for directory in tqdm(self._config.data_directories, ascii=True, desc=__name__):
                run = self.load_run(directory, input_sizes=input_sizes, output_sizes=output_sizes)
                if len(run) != 0:
                    self._dataset.data.append(run)
            cprint(f'Loaded {len(self._dataset.data)} from {len(self._config.data_directories)} directories',
                   self._logger, msg_type=MessageType.error if len(self._dataset.data) == 0 else MessageType.info)

        self.count_datapoints()
        self._num_runs = len(self._dataset.data)
        run_lengths = [len(r) for r in self._dataset.data]
        self._run_probabilities = [r/sum(run_lengths) for r in run_lengths]
        if self._config.balance_targets:
            for run in self._dataset.data:
                self._probabilities_per_run.append(calculate_probabilites_per_run(run))

    def get_data(self) -> list:
        return self._dataset.data

    def load_run(self, directory: str, input_sizes: List[List] = None, output_sizes: List[List] = None) -> Run:
        run = Run()
        time_stamps = {}
        for x_i, x in enumerate(self._config.inputs):
            try:
                time_stamps[x], run.inputs[x] = load_data(x, directory,
                                                          size=input_sizes[x_i] if input_sizes is not None else ())
            except IndexError:
                cprint(f'data loader input config {self._config.inputs} mismatches models input sizes {input_sizes}.',
                       self._logger, msg_type=MessageType.error)
                raise IndexError
        for y_i, y in enumerate(self._config.outputs):
            time_stamps[y], run.outputs[y] = load_data(y, directory,
                                                       size=output_sizes[y_i] if output_sizes is not None else ())
        if self._config.reward:
            time_stamps['reward'], run.reward = load_data(self._config.reward, directory)
        return arrange_run_according_timestamps(run, time_stamps)

    def count_datapoints(self):
        # TODO: add extra time dimension (concat frames & sequence)
        self._num_datapoints = sum([1 for run in self._dataset.data for sample in list(run.inputs.values())[0]])

    def _get_run_length(self, run_index: int) -> int:
        return len(list(self._dataset.data[run_index].inputs.values())[0])

    def _append_datapoint(self, destination: Run, run_index: int, sample_index: int) -> Run:
        for x in self._dataset.data[run_index].inputs.keys():
            destination.inputs[x] = torch_append(destination.inputs[x],
                                                 self._dataset.data[run_index].inputs[x][sample_index].unsqueeze_(0))
        for y in self._dataset.data[run_index].outputs.keys():
            destination.outputs[y] = torch_append(destination.outputs[y],
                                                  self._dataset.data[run_index].outputs[y][sample_index].unsqueeze_(0))
        if self._dataset.data[run_index].reward.size() != (0,):
            destination.reward = torch_append(destination.reward,
                                              self._dataset.data[run_index].reward[sample_index].unsqueeze_(0))
        return destination

    def _get_empty_run(self) -> Run:
        run = Run()
        for x in self._dataset.data[0].inputs.keys():
            run.inputs[x] = torch.Tensor()
        for y in self._dataset.data[0].outputs.keys():
            run.outputs[y] = torch.Tensor()
        if self._dataset.data[0].reward.size() != (0,):
            run.reward = torch.Tensor()
        return run

    def sample_shuffled_batch(self, batch_size: int = 64, max_number_of_batches: int = 1000) -> Generator[Run,
                                                                                                          None,
                                                                                                          None]:
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
        # Calculate sampling weights according to:
        # TODO   d. previous error (prioritized sweeping)
        number_of_batches = min(max_number_of_batches, int(self._num_datapoints / batch_size))
        for batch_index in range(number_of_batches):
            batch = self._get_empty_run()
            for sample in range(batch_size):
                run_index = int(np.random.choice(list(range(self._num_runs)), p=self._run_probabilities))
                sample_index = int(np.random.choice(list(range(self._get_run_length(run_index=run_index))),
                                                    p=self._probabilities_per_run[run_index]))
                batch = self._append_datapoint(destination=batch, run_index=run_index, sample_index=sample_index)
            if len(batch) != 0:
                yield batch
        return
