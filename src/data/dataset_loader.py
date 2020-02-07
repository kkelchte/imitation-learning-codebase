"""Load dataset from local NFS file system and prepares it as pytorch dataset.

"""
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
from dataclasses_json import dataclass_json

from src.core.config_loader import Config
from src.data.data_types import Dataset, Run
from src.data.utils import load_and_preprocess_file, filename_to_timestamp, torch_append


@dataclass_json
@dataclass
class DataLoaderConfig(Config):
    data_directories: List[str] = None
    inputs: List[str] = None
    outputs: List[str] = None
    reward: str = ''

    def __post_init__(self):
        if self.inputs is None:
            self.inputs = ['forward_camera']
        if self.outputs is None:
            self.outputs = ['ros_expert']


class DataLoader:

    def __init__(self, config: DataLoaderConfig):
        self._config = config

    def load(self, size: tuple = ()) -> Dataset:
        dataset = Dataset()
        for directory in self._config.data_directories:
            run = self.load_run(directory, size=size)
            dataset.data.append(run)
        return dataset

    def load_run(self, directory: str, size: tuple = ()) -> Run:
        run = Run()
        time_stamps = {}
        for x in self._config.inputs:
            time_stamps[x], run.inputs[x] = load_data(x, directory, size=size)
        for y in self._config.outputs:
            time_stamps[y], run.outputs[y] = load_data(y, directory)
        if self._config.reward:
            time_stamps['reward'], run.reward = load_data(self._config.reward, directory)
        return arrange_run_according_timestamps(run, time_stamps)


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
        return load_data_from_file(os.path.join(directory, dataype))
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


def load_data_from_file(filename: str) -> Tuple[list, torch.Tensor]:
    with open(filename, 'r') as f:
        lines = f.readlines()
    time_stamps = []
    data = []
    for line in lines:
        time_stamp, data_stamp = line.strip().split(':')
        time_stamps.append(float(time_stamp))
        data.append([float(d) for d in data_stamp.strip().split(' ')])

    return time_stamps, torch.Tensor(data)
