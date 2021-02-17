#!/usr/bin/python3.8

# Helper functions
import json

import torch


def generalized_advantage_estimate(batch_rewards: list,
                                   batch_done: list,
                                   batch_values: list,
                                   discount: float,
                                   gae_lambda: float) -> list:
    advantages = [0] * len(batch_rewards)
    not_done = [1 - d for d in batch_done]  # done is array indicating 'last' frames
    #  not_done array: if value is done future advantage should not influence.
    # the last advantage = last reward + gamma * V_bs * not_done_boolean - last value
    advantages[-1] = batch_rewards[-1] - batch_values[-1]
    for t in reversed(range(len(batch_rewards) - 1)):
        delta = batch_rewards[t] + discount * batch_values[t + 1] * not_done[t] - batch_values[t]
        advantages[t] = delta + discount * gae_lambda * not_done[t] * advantages[t + 1]
    return advantages


def reward_to_go(rewards: list) -> list:
    reward_to_gos = [0] * len(rewards)
    for t in reversed(range(len(rewards))):
        reward_to_gos[t] = rewards[t] + (reward_to_gos[t + 1] if t + 1 < len(rewards) else 0)
    return list(reward_to_gos)


def store_checkpoint(network: torch.nn.Module, file_path_without_extension: str):
    torch.save(network.state_dict(), file_path_without_extension + '.ckpt')

    input_dimension = network[0].in_features
    traced_network = torch.jit.trace_module(network, {'forward': torch.rand(input_dimension)})
    traced_network.save(file_path_without_extension + '.traced')


def add_info_to_logger(key: str, value: float, logger: dict) -> dict:
    if key in logger.keys():
        logger[key].append(value)
    else:
        logger[key] = [value]
    return logger


def store_logger_results(logger: dict, output_path: str) -> None:
    with open(f'{output_path}/logger_results.json', 'w') as f:
        json.dump(logger, f)
