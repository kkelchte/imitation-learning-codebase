from typing import Union, List, Iterator

import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter

from src.core.data_types import Dataset, to_torch
from src.data.data_saver import DataSaverConfig, DataSaver
from src.data.test.common_utils import generate_dummy_dataset


def initialize_weights(weights: torch.nn.Module, initialisation_type: str = 'xavier', scale: float = 2**0.5) -> None:
    for p in weights.parameters():
        if len(p.shape) <= 1:
            p.data.zero_()
        else:
            if initialisation_type == 'xavier':
                nn.init.xavier_uniform_(p.data)
            elif initialisation_type == 'constant':
                nn.init.constant_(p.data, scale)
            elif initialisation_type == 'orthogonal':
                nn.init.orthogonal_(p.data, scale)
            else:
                raise NotImplementedError


class DiscreteActionMapper:

    def __init__(self, action_values: List[torch.Tensor]):
        self.action_values = action_values

    def tensor_to_index(self, action_as_tensor: torch.Tensor) -> int:
        for index, action in enumerate(self.action_values):
            if (action - action_as_tensor).sum() < 0.001:
                return index
        raise SyntaxError(f'Could not map action {action_as_tensor}')

    def index_to_tensor(self, action_as_index: int) -> torch.Tensor:
        return self.action_values[action_as_index]


def data_to_tensor(data: Union[list, np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Prepare data as tensor"""
    try:
        data = torch.as_tensor(data)
    except ValueError:
        data = torch.stack(data)
    return data


def mlp_creator(sizes: List[int], activation: nn.Module, output_activation: nn.Module = None):
    """Create Multi-Layer Perceptron"""
    layers = []
    for j in range(len(sizes)-1):
        is_not_last_layer = j < len(sizes)-2
        layers += [nn.Linear(sizes[j], sizes[j+1], bias=is_not_last_layer)]
        act = activation if is_not_last_layer else output_activation
        if act is not None:
            layers += [act()]
    return nn.Sequential(*layers)

##################################################################
# Test Helper Functions
##################################################################


def generate_random_dataset_in_raw_data(output_dir: str, num_runs: int = 20,
                                        input_size: tuple = (100, 100, 3), output_size: tuple = (1,),
                                        continuous: bool = True,
                                        fixed_output_value: Union[float, np.ndarray] = None,
                                        store_hdf5: bool = False) -> dict:
    """Generate data, stored in raw_data directory of output_dir"""
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


def get_checksum_network_parameters(parameters: Union[List[torch.Tensor],
                                                      Iterator[Parameter]]) -> float:
    count_weights = 0
    for p in parameters:
        count_weights += torch.sum(p).item()
    return count_weights

##################################################################
# Phi (gradient weight) estimates
##################################################################


def discount_path(path, h):
    '''
    Given a "path" of items x_1, x_2, ... x_n, return the discounted
    path, i.e.
    X_1 = x_1 + h*x_2 + h^2 x_3 + h^3 x_4
    X_2 = x_2 + h*x_3 + h^2 x_4 + h^3 x_5
    etc.
    Can do (more efficiently?) w SciPy. Python here for readability
    Inputs:
    - path, list/tensor of floats
    - h, discount rate
    Outputs:
    - Discounted path, as above
    '''
    curr = 0
    rets = []
    for i in range(len(path)):
        curr = curr*h + path[-1-i]
        rets.append(curr)
    rets =  torch.stack(list(reversed(rets)), 0)
    return rets


def get_path_indices(not_dones):
    """
    Returns list of tuples of the form:
        (agent index, time index start, time index end + 1)
    For each path seen in the not_dones array of shape (# agents, # time steps)
    E.g. if we have an not_dones of composition:
    tensor([[1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 0, 1, 1, 0, 1]], dtype=torch.uint8)
    Then we would return:
    [(0, 0, 3), (0, 3, 10), (1, 0, 3), (1, 3, 5), (1, 5, 9), (1, 9, 10)]
    """
    indices = []
    num_timesteps = not_dones.shape[1]
    for actor in range(not_dones.shape[0]):
        last_index = 0
        for i in range(num_timesteps):
            if not_dones[actor, i] == 0.:
                indices.append((actor, last_index, i + 1))
                last_index = i + 1
        if last_index != num_timesteps:
            indices.append((actor, last_index, num_timesteps))
    return indices


def get_generalized_advantage_estimate_with_tensors(batch_rewards: torch.Tensor,
                                                    batch_done: torch.Tensor,
                                                    batch_values: torch.Tensor,
                                                    discount: float = 0.99,
                                                    gae_lambda: float = 0.95) -> torch.Tensor:
    batch_not_done = -1 * batch_done.bool() + 1
    V_s_tp1 = torch.cat([batch_values[1:], batch_values[-1:]], 0) * batch_not_done.squeeze()
    deltas = batch_rewards.squeeze_() + gae_lambda * V_s_tp1 - batch_values

    # now we need to discount each path by gamma * lam
    advantages = torch.zeros_like(batch_rewards)
    indices = get_path_indices(batch_not_done.unsqueeze(0))
    for _, start, end in indices:
        advantages[start:end] = discount_path(deltas[start:end], gae_lambda * discount)
    # advantages = torch.zeros(batch_rewards.shape)
    # #  not_done array: if value is done future advantage should not influence.
    # # the last advantage = last reward + gamma * V_bs * not_done_boolean - last value
    # advantages[-1] = batch_rewards[-1] - batch_values[-1]
    # for t in reversed(range(len(batch_rewards) - 1)):
    #     delta = batch_rewards[t] + discount * batch_values[t + 1] * batch_not_done[t] - batch_values[t]
    #     advantages[t] = delta + discount * gae_lambda * batch_not_done[t] * advantages[t + 1]
    return advantages


def get_generalized_advantage_estimate(batch_rewards: List[torch.Tensor],
                                       batch_done: List[torch.Tensor],
                                       batch_values: List[torch.Tensor],
                                       discount: float,
                                       gae_lambda: float) -> torch.Tensor:
    batch_done = torch.stack(batch_done) if isinstance(batch_done, list) else batch_done
    batch_not_done = -1 * batch_done.bool() + 1
#    try:
#        batch_not_done = [torch.as_tensor(1) if d.data == 0 else torch.as_tensor(0) for d in batch_done]
#    except:
#        batch_not_done = [torch.as_tensor(1) if not d else torch.as_tensor(0) for d in batch_done]
    advantages = [torch.as_tensor(0)] * len(batch_rewards)
    #  not_done array: if value is done future advantage should not influence.
    # the last advantage = last reward + gamma * V_bs * not_done_boolean - last value
    advantages[-1] = batch_rewards[-1] - batch_values[-1]
    for t in reversed(range(len(batch_rewards) - 1)):
        delta = batch_rewards[t] + discount * batch_values[t + 1] * batch_not_done[t] - batch_values[t]
        advantages[t] = delta + discount * gae_lambda * batch_not_done[t] * advantages[t + 1]
    return torch.as_tensor(advantages)


def get_reward_to_go(batch: Dataset) -> torch.Tensor:
    returns = [torch.as_tensor(0)] * len(batch)
    for t in reversed(range(len(batch))):
        if t + 1 < len(batch) and batch.done[t] == 0:
            returns[t] = batch.rewards[t] + returns[t + 1]
        else:
            returns[t] = batch.rewards[t]
    assert len(returns) == len(batch)
    return torch.as_tensor(returns)


def get_returns(batch: Dataset) -> torch.Tensor:
    returns = []
    count_steps = 0
    count_rewards = 0
    for done, reward in zip(batch.done, batch.rewards):
        count_rewards += reward
        count_steps += 1
        if done != 0:
            returns.extend([to_torch(count_rewards)] * count_steps)
            count_steps = 0
            count_rewards = 0

    assert len(returns) == len(batch)
    return torch.as_tensor(returns)
