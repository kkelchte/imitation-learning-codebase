from typing import Union, List

import torch
import torch.nn as nn
import numpy as np

from src.data.data_saver import DataSaverConfig, DataSaver
from src.data.test.common_utils import generate_dummy_dataset


def process_input(self, inputs: Union[List[torch.Tensor],
                                      torch.Tensor,
                                      List[np.ndarray],
                                      np.ndarray]) -> torch.Tensor:
    if not isinstance(inputs, list):
        inputs = [inputs]
    processed_inputs = []
    for shape, data in zip(self._architecture.input_sizes, inputs):
        assert (isinstance(data, np.ndarray) or isinstance(data, torch.Tensor))
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, dtype=torch.float32)
        if np.argmin(data.size()) != 0:  # assume H,W,C --> C, H, W
            data = data.permute(2, 0, 1)
        processed_inputs.append(data.reshape(shape).to(self._device))
    return torch.stack(processed_inputs, dim=0)


def mlp_creator(sizes: List[int], activation: nn.Module, output_activation=nn.Module):
    layers = []
    for j in range(len(sizes)-1):
        is_not_last_layer = j < len(sizes)-2
        layers += [nn.Linear(sizes[j], sizes[j+1], bias=is_not_last_layer)]
        act = activation if is_not_last_layer else output_activation
        if act is not None:
            layers += [act()]
    return nn.Sequential(*layers)


def generate_random_dataset_in_raw_data(output_dir: str, num_runs: int = 20,
                                        input_size: tuple = (100, 100, 3), output_size: tuple = (1,),
                                        continuous: bool = True,
                                        fixed_output_value: float = None,
                                        store_hdf5: bool = False) -> dict:
    data_saver = DataSaver(config=DataSaverConfig().create(config_dict={'output_path': output_dir,
                                                                        'store_hdf5': store_hdf5}))
    info = generate_dummy_dataset(data_saver,
                                  num_runs=num_runs,
                                  input_size=input_size,
                                  output_size=output_size,
                                  continuous=continuous,
                                  fixed_output_value=fixed_output_value,
                                  store_hdf5=store_hdf5)
    return info
