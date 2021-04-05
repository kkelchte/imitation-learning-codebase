from typing import Union, List, Iterator, Sequence, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter

from src.core.data_types import Dataset, to_torch
from src.data.data_saver import DataSaverConfig, DataSaver
from src.data.test.common_utils import generate_dummy_dataset


def clip_action_according_to_playfield_size_flipped(inputs: Union[torch.Tensor, np.ndarray],
                                            actions: Union[torch.Tensor, np.ndarray],
                                            playfield_size: Tuple[float, float, float]) -> np.ndarray:
    """
    clip velocities of actions to zero when agents location is further from origin than playfield size.
    inputs: dimension (9,) [tracking_x, tracking_y, tracking_z,
                            fleeing_x, fleeing_y, fleeing_z,
                            tracking_roll, tracking_pitch, tracking_yaw]
    actions: dimension (8,) [tracking_linear_x, tracking_linear_y, tracking_linear_z,
                             fleeing_linear_x, fleeing_linear_y, fleeing_linear_z,
                             tracking_angular_z, fleeing_angular_z]
     playfield_size: tuple (3,) specifying boundaries of playfield in x, y, z
    """
    assert inputs.shape == (9,)
    assert actions.shape == (8,)
    assert len(playfield_size) == 3
    for agent in ['tracking', 'fleeing']:
        for direction_index in range(3):
            if playfield_size[direction_index] == 0:
                continue
            index = direction_index if agent == 'tracking' else direction_index + 3
            position = inputs[index]
            if abs(position) > playfield_size[direction_index]:
                if position > 0:  # clip positive action to zero
                    if agent == 'fleeing' and index < 5:
                        actions[index] = max(0, actions[index])
                    else:
                        actions[index] = min(0, actions[index])
                else:  # clip negative action to zero
                    if agent == 'fleeing' and index < 5:
                        actions[index] = min(0, actions[index])
                    else:
                        actions[index] = max(0, actions[index])
    return actions


def clip_action_according_to_playfield_size(inputs: Union[torch.Tensor, np.ndarray],
                                            actions: Union[torch.Tensor, np.ndarray],
                                            playfield_size: Tuple[float, float, float]) -> np.ndarray:
    """
    clip velocities of actions to zero when agents location is further from origin than playfield size.
    inputs: dimension (9,) [tracking_x, tracking_y, tracking_z,
                            fleeing_x, fleeing_y, fleeing_z,
                            tracking_roll, tracking_pitch, tracking_yaw]
    actions: dimension (8,) [tracking_linear_x, tracking_linear_y, tracking_linear_z,
                             fleeing_linear_x, fleeing_linear_y, fleeing_linear_z,
                             tracking_angular_z, fleeing_angular_z]
     playfield_size: tuple (3,) specifying boundaries of playfield in x, y, z
    """
    assert inputs.shape == (9,)
    assert actions.shape == (8,)
    assert len(playfield_size) == 3
    for agent in ['tracking', 'fleeing']:
        for direction_index in range(3):
            if playfield_size[direction_index] == 0:
                continue
            index = direction_index if agent == 'tracking' else direction_index + 3
            position = inputs[index]
            if abs(position) > playfield_size[direction_index]:
                if position > 0:  # clip positive action to zero
                    actions[index] = min(0, actions[index])
                else:  # clip negative action to zero
                    actions[index] = max(0, actions[index])
    return actions


def get_slow_hunt(state: torch.Tensor) -> torch.Tensor:
    agent_zero = state[:2]
    agent_one = state[2:]
    return 0.3 * np.sign(agent_one - agent_zero)


def get_slow_run(state: np.ndarray) -> np.ndarray:
    agent_zero = state[:2]
    agent_one = state[2:]
    difference = (agent_one - agent_zero)
    for diff in difference:
        if diff == 0:
            difference += (np.random.rand(2) - 0.5) / 10
    difference = np.sign(difference)
    return 0.2 * difference


def get_waypoint(playfield_size: Tuple[float, float, float]) -> np.ndarray:
    assert len(playfield_size) == 3
    return np.multiply(2 * np.asarray(playfield_size), np.random.random(3)) - np.asarray(playfield_size) \
        + np.asarray([3, 0, 0])


def get_rand_run_ros(waypoint: np.ndarray, position: np.ndarray, playfield_size: Tuple[float, float, float]) \
        -> np.ndarray:
    action = [0, 0, 0]
    for i in range(3):
        if playfield_size[i] == 0:
            continue
        if abs(waypoint[i] - position[i]) < 0.1:
            waypoint = get_waypoint(playfield_size)
        action[i] = -0.3*np.sign(waypoint[i] - position[i])
    return np.asarray([waypoint, action])


def get_slow_run_ros(state: np.ndarray, playfield_size: Tuple[float, float, float]) -> np.ndarray:
    x_center = 200
    x_flee = state[0]
    action = [0, 0, 0]
    for i in range(3):
        if playfield_size[i] == 0:
            continue
        action[i] = 0.3*np.sign(x_flee - x_center)
    return np.asarray(action)


def get_slow_hunt_ros(state: np.ndarray, playfield_size: Tuple[float, float, float]) -> np.ndarray:
    x_center = 200
    x_flee = state[0]
    action = [0, 0, 0]
    for i in range(3):
        if playfield_size[i] == 0:
            continue
        action[i] = -0.3 * np.sign(x_flee - x_center)
    return np.asarray(action)


def initialize_weights(weights: torch.nn.Module, initialisation_type: str = 'xavier', scale: float = 2 ** 0.5) -> None:
    for p in weights.parameters():
        if len(p.shape) == 1:
            p.data.zero_()
        elif len(p.shape) > 1:
            if initialisation_type == 'xavier':
                nn.init.xavier_uniform_(p.data)
            elif initialisation_type == 'constant':
                nn.init.constant_(p.data, 0.03)
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


def mlp_creator(sizes: List[int], activation: nn.Module = None, output_activation: nn.Module = None,
                bias_in_last_layer: bool = True) -> nn.Module:
    """Create Multi-Layer Perceptron"""
    layers = []
    for j in range(len(sizes) - 1):
        is_not_last_layer = j < len(sizes) - 2
        layers += [nn.Linear(sizes[j], sizes[j + 1], bias=True if bias_in_last_layer else is_not_last_layer)]
        act = activation if is_not_last_layer else output_activation
        if act is not None:
            layers += [act]
    return nn.Sequential(*layers)


def conv_creator(channels: List[int], kernel_sizes: List[int] = None,
                 strides: List[int] = None, activation: nn.Module = None,
                 output_activation: nn.Module = None,
                 bias_in_last_layer: bool = True,
                 batch_norm: bool = False) -> nn.Module:
    """Create Conv2d Network"""
    layers = []
    for j in range(len(channels) - 1):
        is_not_last_layer = j < len(channels) - 2
        layers += [nn.Conv2d(in_channels=channels[j],
                             out_channels=channels[j + 1],
                             kernel_size=3 if kernel_sizes is None else kernel_sizes[j],
                             stride=2 if strides is None else strides[j],
                             bias=True if bias_in_last_layer else is_not_last_layer)]
        if batch_norm:
            layers += [nn.BatchNorm2d(channels[j + 1])]
        act = activation if is_not_last_layer else output_activation
        if act is not None:
            layers += [act]
    return nn.Sequential(*layers)


##################################################################
# Test Helper Functions
##################################################################


def generate_random_dataset_in_raw_data(output_dir: str, num_runs: int = 20,
                                        input_size: tuple = (100, 100, 3), output_size: tuple = (1,),
                                        continuous: bool = True,
                                        fixed_input_value: Union[float, np.ndarray] = None,
                                        fixed_output_value: Union[float, np.ndarray] = None,
                                        store_hdf5: bool = False) -> dict:
    """Generate data, stored in raw_data directory of output_dir"""
    data_saver = DataSaver(config=DataSaverConfig().create(config_dict={'output_path': output_dir,
                                                                        'store_hdf5': store_hdf5,
                                                                        'separate_raw_data_runs': True}))
    info = generate_dummy_dataset(data_saver,
                                  num_runs=num_runs,
                                  input_size=input_size,
                                  output_size=output_size,
                                  continuous=continuous,
                                  fixed_input_value=fixed_input_value,
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


def get_generalized_advantage_estimate(batch_rewards: List[torch.Tensor],
                                       batch_done: List[torch.Tensor],
                                       batch_values: torch.Tensor,
                                       discount: float,
                                       gae_lambda: float) -> torch.Tensor:
    batch_done = torch.stack(batch_done) if isinstance(batch_done, list) else batch_done
    batch_not_done = -1 * batch_done.bool() + 1
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
