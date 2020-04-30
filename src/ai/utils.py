from typing import Union, List

import torch
import torch.nn as nn
import numpy as np

from src.data.data_saver import DataSaverConfig, DataSaver
from src.data.test.common_utils import generate_dummy_dataset


def data_to_tensor(data: Union[list, np.ndarray, torch.Tensor]) -> torch.Tensor:
    try:
        data = torch.as_tensor(data)
    except ValueError:
        data = torch.stack(data)
    return data


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
