from typing import Union, List

import torch
import torch.nn as nn
import numpy as np


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
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def generate_random_dataset(output_dir: str, input_size: tuple, output_size: tuple) -> bool:
    return True