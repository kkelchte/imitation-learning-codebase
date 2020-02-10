import os

import torch
from PIL import Image
import numpy as np


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
        assert np.amax(data) < 1 and np.amin(data) > 0
        if grayscale:
            data = data.mean(axis=-1, keepdims=True)
        data = data.swapaxes(1, 2).swapaxes(0, 1)  # make channel first
    else:
        raise NotImplementedError
    return data


def torch_append(destination: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    if destination.size() != (0,):
        assert destination[0].size() == source[0].size()
    return torch.cat((destination, source), 0)
