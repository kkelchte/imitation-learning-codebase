import os
from typing import List, Tuple

import h5py
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

from src.data.data_types import Run, Dataset


def timestamp_to_filename(time_stamp_ms: int) -> str:
    return f'{time_stamp_ms:015d}'


def filename_to_timestamp(filename: str) -> int:
    return int(os.path.basename(filename).split('.')[0])


def store_image(data: np.ndarray, file_name: str) -> None:
    assert np.amin(data) >= 0
    assert np.amax(data) <= 1
    assert data.dtype == np.float32 or data.dtype == np.float16 or data.dtype == np.float64

    processed_data = (data * 255).astype(np.uint8)
    im = Image.fromarray(processed_data)
    im.save(file_name)  # await


def store_array_as_numpy(data: np.ndarray, file_name: str) -> None:
    np.save(file_name, data)


def store_array_to_file(data: np.ndarray, file_name: str, time_stamp: int = 0) -> None:
    message = f'{time_stamp} : ' + ' '.join(f'{x:0.5f}' for x in data) + '\n'
    with open(file_name, 'a') as f:
        f.write(message)


def load_and_preprocess_file(file_name: str, sensor_name: str, size: tuple = (), grayscale: bool = False) -> np.ndarray:
    if 'camera' in sensor_name:
        data = Image.open(file_name, mode='r')
        if size:
            # assume [channel, height, width]
            assert min(size) == size[0]
            data = data.resize(size[1:])
        data = np.array(data).astype(np.float32)  # uint8 -> float32
        data /= 255.  # 0:255 -> 0:1
        assert np.amax(data) <= 1 and np.amin(data) >= 0
        if grayscale:
            data = data.mean(axis=-1, keepdims=True)
        data = data.swapaxes(1, 2).swapaxes(0, 1)  # make channel first
    else:
        raise NotImplementedError
    return data


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


def torch_append(destination: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    if destination.size() != (0,):
        assert destination[0].size() == source[0].size()
    return torch.cat((destination, source), 0)


def add_run_to_h5py(h5py_file: h5py.File, run: str) -> h5py.File:
    data = {}
    time_stamps = {}
    # load all data from run
    for element_name in os.listdir(run):
        if element_name == 'termination':
            continue
        element = os.path.join(run, element_name)
        time_stamps[element_name], data[element_name] = load_data_from_file(element) if os.path.isfile(element) else \
            load_data_from_directory(element)
    if len(data.keys()) == 0:
        return h5py_file
    # arrange all data to have corresponding time stamps
    clean_data = {
        element_name: torch.Tensor() for element_name in data.keys()
    }
    while min([len(time_stamps[element_name]) for element_name in time_stamps.keys()]) != 0:
        # get first coming time stamp
        current_time_stamp = min([time_stamps[element_name][0] for element_name in time_stamps.keys()])
        # check if all inputs & outputs & rewards have a value for this stamp
        check = True
        for element_name in data.keys():
            check = time_stamps[element_name][0] == current_time_stamp and check
        if check:
            for element_name in data.keys():
                clean_data[element_name] = torch_append(clean_data[element_name], data[element_name][0].unsqueeze_(0))
        for element_name in data.keys():
            while len(time_stamps[element_name]) != 0 and time_stamps[element_name][0] == current_time_stamp:
                data[element_name] = data[element_name][1:] if len(data[element_name]) > 1 else []
                time_stamps[element_name] = time_stamps[element_name][1:] if len(time_stamps[element_name]) > 1 else []
    del data
    # final clean up check
    lengths = [len(value) for value in clean_data.values()]
    assert min(lengths) == max(lengths)
    if lengths != 0:
        run_group = h5py_file.create_group(os.path.basename(run))
        # each data element becomes a separate dataset in the h5py_file
        for element_name in clean_data.keys():
            run_group[element_name] = clean_data[element_name].numpy()
    return h5py_file


def create_hdf5_file(filename: str, runs: List[str]) -> None:
    h5py_file = h5py.File(filename, 'w')
    for run in tqdm(runs, ascii=True, desc=f'creating {os.path.basename(filename)}'):
        h5py_file = add_run_to_h5py(h5py_file=h5py_file, run=run)


def load_dataset_from_hdf5(filename: str, inputs: List[str], outputs: List[str]) -> Dataset:
    dataset = Dataset()
    h5py_file = h5py.File(filename, 'r')
    for h5py_group in tqdm(h5py_file, ascii=True, desc=f'load {os.path.basename(filename)}'):
        run = load_run_from_h5py(h5py_file[h5py_group], inputs, outputs)
        if len(run) != 0:
            dataset.data.append(run)
    return dataset


def load_run_from_h5py(group: h5py.Group, inputs: List[str], outputs: List[str]) -> Run:
    run = Run()
    for field_name in group:
        if field_name in inputs:
            run.inputs[field_name] = torch.Tensor(group[field_name])
        if field_name in outputs:
            run.outputs[field_name] = torch.Tensor(group[field_name])
    return run
