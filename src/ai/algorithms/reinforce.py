#!/usr/bin/python3.7
import os
import sys
import time
import gym
import numpy as np
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.optim import Adam
import matplotlib.pyplot as plt

from src.ai.algorithms.utils import *
from src.ai.architectures.cart_pole_4_2d_stochastic import Net
##############################################################
# Settings
from src.ai.algorithms.utils import generalized_advantage_estimate, reward_to_go
from src.ai.base_net import ArchitectureConfig
from src.ai.trainer import TrainerConfig
from src.ai.utils import get_checksum_network_parameters
from src.ai.vpg import VanillaPolicyGradient
from src.core.data_types import Dataset
from src.core.utils import get_check_sum_list

result_file_name = 'reinforce_GAE_cartpole_v0'
weight_type = 'reward_to_go_with_generalized_advantage_estimate'
environment_name = 'CartPole-v0'  # 'Acrobot-v1'  # 'CartPole-v1'
learning_rate = 1e-3
epochs = 1000
batch_size = 200
discount = 0.95
gae_lambda = 0.95
plot = True
seed = 123
##############################################################
# Global variables:
environment = gym.make(environment_name)
environment.seed(seed)
discrete = isinstance(environment.action_space, gym.spaces.Discrete)

observation_dimension = environment.observation_space.shape[0]
action_dimension = environment.action_space.n if discrete else environment.action_space.shape[0]
# policy_network = nn.Sequential(
#     nn.Linear(observation_dimension, 20), nn.ReLU(inplace=True),
#     nn.Linear(20, 20), nn.ReLU(inplace=True),
#     nn.Linear(20, 20), nn.ReLU(inplace=True),
#     nn.Linear(20, action_dimension),
# )
# if not discrete:
#     log_std = torch.nn.Parameter(torch.as_tensor(-0.5 * np.ones(action_dimension, dtype=np.float32)))
#

# value_network = nn.Sequential(
#     nn.Linear(observation_dimension, 20), nn.ReLU(inplace=True),
#     nn.Linear(20, 20), nn.ReLU(inplace=True),
#     nn.Linear(20, 20), nn.ReLU(inplace=True),
#     nn.Linear(20, 1)
# )
# for network in [policy_network, value_network]:
#     for p in network.parameters():
#         if len(p.shape) == 1:
#             nn.init.uniform_(p, a=0, b=1)
#         else:
#             nn.init.xavier_uniform_(p)

#
torch.manual_seed(seed)
net = Net(config=ArchitectureConfig().create(config_dict={
    'output_path': os.path.join(os.getcwd(), 'output_dir', 'reinforce'),
    'architecture': 'cart_pole_4_2d_stochastic',
    'device': 'cpu',
    'initialisation_seed': 123,
    'initialisation_type': 0,
}))

policy_network = net._actor
value_network = net._critic
print(get_checksum_network_parameters(list(policy_network.parameters())))
print(get_checksum_network_parameters(list(value_network.parameters())))
# optimizer = Adam(net.get_actor_parameters(), lr=learning_rate)
# value_optimizer = Adam(net.get_critic_parameters(), lr=learning_rate)

trainer = VanillaPolicyGradient(config=TrainerConfig().create(config_dict={
                                        'output_path': os.path.join(os.getcwd(), 'output_dir', 'reinforce'),
                                        "criterion": "MSELoss",
                                        "device": "cpu",
                                        "learning_rate": 0.001,
                                        "optimizer": "Adam",
                                        "data_loader_config": {"batch_size": 200},
                                        "discount": 0.95,
                                        "factory_key": "VPG",
                                        "gae_lambda": 0.95,
                                        "phi_key": "gae",
                                        "save_checkpoint_every_n": 1000
                                }), network=net)


def policy_forward(observation):
    return net._policy_distribution(observation, train=True)
    # logits = policy_network(observation)
    # if discrete:
    #     return Categorical(logits=logits)
    # else:
    #     return Normal(logits, torch.exp(log_std))


def evaluate_policy(observation):
    output = policy_forward(torch.as_tensor(observation, dtype=torch.float32)).sample()
    if discrete:
        return output.item()
    else:
        return output


def train_one_epoch():
    batch_weights = []
    batch_observations = []
    batch_actions = []
    batch_rewards = []
    batch_done = []
    epoch_returns = []
    episode_lengths = []

    observation = environment.reset()
    done = False
    episode_rewards = []
    while True:
        batch_observations.append(observation.copy())
        #action = evaluate_policy(observation)
        action = net.get_action(observation, train=False).value
        observation, reward, done, info = environment.step(action)
        episode_rewards.append(reward)
        batch_actions.append(action)
        batch_done.append(done)
        batch_rewards.append(reward)
        if done:
            episode_return = sum(episode_rewards)
            episode_length = len(episode_rewards)
            if 'reward_to_go' in weight_type:
                batch_weights += list(reward_to_go(episode_rewards))
            else:
                batch_weights += [episode_return] * episode_length
            epoch_returns.append(episode_return)
            episode_lengths.append(episode_length)
            observation = environment.reset()
            done = False
            episode_rewards = []

            if len(batch_observations) > batch_size:
                break

    batch = Dataset(
        observations=[torch.as_tensor(o) for o in batch_observations],
        actions=[torch.as_tensor(x, dtype=torch.float32) for x in batch_actions],
        rewards=[torch.as_tensor(x, dtype=torch.float32) for x in batch_rewards],
        done=[torch.as_tensor(x, dtype=torch.float32) for x in batch_done]
    )

    trainer.data_loader.set_dataset(batch)
    trainer.train(epoch=1)
    batch_loss = 0
    value_loss = 0

    return batch_loss, value_loss, epoch_returns

############################################
# Loop over epochs and keep returns


start_time = time.time()

avg_returns = []
min_returns = []
max_returns = []

best_avg_return = -10000
result_directory = f'{os.environ["HOME"]}/code/imitation-learning-codebase/src/ai/algorithms/' \
                   f'results/reinforce'
os.makedirs(result_directory, exist_ok=True)

for i in range(epochs):
    batch_loss, value_loss, batch_returns = train_one_epoch()
    avg_returns.append(np.mean(batch_returns))
    min_returns.append(min(batch_returns))
    max_returns.append(max(batch_returns))
    print(f'epoch: {i} \t actor loss: {batch_loss:0.3f} \t '
          f'returns: {np.mean(batch_returns): 0.3f} [{np.std(batch_returns): 0.3f}] \t'
          f'value loss: {value_loss: 0.3f}')
    if (i % 20 == 10 or i == epochs - 1) and plot:
        plt.fill_between(range(len(avg_returns)), min_returns, max_returns, color='blue', alpha=0.5)
        plt.plot(avg_returns, color='b')
        plt.show()
    if np.mean(batch_returns) > best_avg_return:
        store_checkpoint(policy_network, result_directory + '/policy')
        best_avg_return = np.mean(batch_returns)

print(f'total duration : {time.time()-start_time}s')
results = np.asarray([avg_returns, min_returns, max_returns])
np.save(os.path.join(result_directory, result_file_name + '.npy'), results)

print(get_checksum_network_parameters(list(policy_network.parameters())))
print(get_checksum_network_parameters(list(value_network.parameters())))
