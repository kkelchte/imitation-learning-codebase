"""Load dataset from local NFS file system and prepares it as pytorch dataset.

"""
import os
from dataclasses import dataclass
from typing import List, Tuple, Generator

import numpy as np
import torch
from dataclasses_json import dataclass_json

from src.core.config_loader import Config
from src.core.logger import cprint, get_logger, MessageType
from src.data.data_types import Dataset, Run
from src.data.utils import load_and_preprocess_file, filename_to_timestamp, torch_append


@dataclass_json
@dataclass
class DataLoaderConfig(Config):
    data_directories: List[str] = None
    inputs: List[str] = None
    outputs: List[str] = None
    reward: str = ''

    def __post_init__(self):  # add default options
        if self.inputs is None:
            self.inputs = ['forward_camera']
        if self.outputs is None:
            self.outputs = ['ros_expert']

    def iterative_add_output_path(self, output_path: str) -> None:
        if self.output_path is None:
            self.output_path = output_path
        if not self.data_directories[0].startswith('/'):
            self.data_directories = [os.path.join(self.output_path, d) for d in self.data_directories]
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                value.iterative_add_output_path(output_path)


class DataLoader:

    def __init__(self, config: DataLoaderConfig):
        self._config = config
        self._logger = get_logger(name=__name__,
                                  output_path=config.output_path,
                                  quite=False)
        cprint(f'Started.', self._logger)
        self._dataset = Dataset()
        self._num_datapoints = 0
        self._num_runs = 0

    def load_dataset(self, input_sizes: List[List] = None, output_sizes: List[List] = None):
        for dir_index, directory in enumerate(self._config.data_directories):
            # TODO: change to tqdm
            if dir_index % 10 == 0:
                cprint(f'loading dir {dir_index}/{len(self._config.data_directories)}')
            run = self.load_run(directory, input_sizes=input_sizes, output_sizes=output_sizes)
            if len(run) != 0:
                self._dataset.data.append(run)
        self.count_datapoints()
        self._num_runs = len(self._dataset.data)

        cprint(f'Loaded {self._num_runs} from {len(self._config.data_directories)} directories', self._logger,
               msg_type=MessageType.error if self._num_runs == 0 else MessageType.info)

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
        # TODO   a. number of samples per run
        # TODO   b. steering balancing
        # TODO   c. speed balancing
        # TODO   d. previous error (prioritized sweeping)
        number_of_batches = min(max_number_of_batches, int(self._num_datapoints / batch_size))
        for batch_index in range(number_of_batches):
            batch = self._get_empty_run()
            for sample in range(batch_size):
                run_index = int(np.random.choice(list(range(self._num_runs)), p=None))
                sample_index = int(np.random.choice(list(range(self._get_run_length(run_index=run_index))), p=None))
                batch = self._append_datapoint(destination=batch, run_index=run_index, sample_index=sample_index)
            if len(batch) != 0:
                yield batch
        return


def arrange_run_according_timestamps(run: Run, time_stamps: dict) -> Run:
    """Ensure there is a data row in the torch tensor for each time stamp.
    """
    clean_run = Run()
    for x in run.inputs.keys():
        clean_run.inputs[x] = torch.Tensor()
        assert len(time_stamps[x]) == len(run.inputs[x])
    for y in run.outputs.keys():
        clean_run.outputs[y] = torch.Tensor()
        assert len(time_stamps[y]) == len(run.outputs[y])

    while min([len(time_stamps[data_type]) for data_type in time_stamps.keys()]) != 0:
        # get first coming time stamp
        current_time_stamp = min([time_stamps[data_type][0] for data_type in time_stamps.keys()])
        # check if all inputs & outputs & rewards have a value for this stamp
        check = True
        for x in run.inputs.keys():
            check = time_stamps[x][0] == current_time_stamp and check
        for y in run.outputs.keys():
            check = time_stamps[y][0] == current_time_stamp and check
        if run.reward.size() != (0,):
            check = time_stamps['reward'][0] == current_time_stamp and check
        if check:  # if check, add tensor to current tensors
            for x in run.inputs.keys():
                clean_run.inputs[x] = torch_append(clean_run.inputs[x], run.inputs[x][0].unsqueeze_(0))
            for y in run.outputs.keys():
                clean_run.outputs[y] = torch_append(clean_run.outputs[y], run.outputs[y][0].unsqueeze_(0))
            if run.reward.size() != (0,):
                clean_run.reward = torch_append(clean_run.reward, run.reward[0].unsqueeze_(0))
        # discard data corresponding to this timestamp
        for x in run.inputs.keys():
            while len(time_stamps[x]) != 0 and time_stamps[x][0] == current_time_stamp:
                run.inputs[x] = run.inputs[x][1:] if len(run.inputs[x]) > 1 else []
                time_stamps[x] = time_stamps[x][1:] if len(time_stamps[x]) > 1 else []
        for y in run.outputs.keys():
            while len(time_stamps[y]) != 0 and time_stamps[y][0] == current_time_stamp:
                run.outputs[y] = run.outputs[y][1:] if len(run.outputs[y]) > 1 else []
                time_stamps[y] = time_stamps[y][1:] if len(time_stamps[y]) > 1 else []
        while run.reward.size() != (0,) and len(time_stamps['reward']) != 0 \
                and time_stamps['reward'][0] == current_time_stamp:
            run.reward = run.reward[1:] if len(run.reward) > 1 else []
            time_stamps['reward'] = time_stamps['reward'][1:] if len(time_stamps['reward']) > 1 else []
    return clean_run


def load_data(dataype: str, directory: str, size: tuple = ()) -> Tuple[list, torch.Tensor]:
    if os.path.isdir(os.path.join(directory, dataype)):
        return load_data_from_directory(os.path.join(directory, dataype), size=size)
    elif os.path.isfile(os.path.join(directory, dataype)):
        return load_data_from_file(os.path.join(directory, dataype), size=size)
    else:
        return [], torch.Tensor()


def load_data_from_directory(directory: str, size: tuple = ()) -> Tuple[list, torch.Tensor]:
    time_stamps = []
    data = []
    for f in sorted(os.listdir(directory)):
        data.append(load_and_preprocess_file(file_name=os.path.join(directory, f),
                                             sensor_name=os.path.basename(directory),
                                             size=size))
        time_stamps.append(filename_to_timestamp(f))
    return time_stamps, torch.Tensor(data)


def load_data_from_file(filename: str, size: tuple = ()) -> Tuple[list, torch.Tensor]:
    with open(filename, 'r') as f:
        lines = f.readlines()
    time_stamps = []
    data = []
    for line in lines:
        time_stamp, data_stamp = line.strip().split(':')
        time_stamps.append(float(time_stamp))
        data_vector = torch.Tensor([float(d) for d in data_stamp.strip().split(' ')])
        if size is not None and size != ():
            data_vector = data_vector.reshape(size)
        data.append(data_vector)
    return time_stamps, torch.stack(data)
