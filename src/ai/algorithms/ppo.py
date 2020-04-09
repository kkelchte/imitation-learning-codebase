#!/usr/bin/python3.7
import os
import time
import gym
import numpy as np
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.optim import Adam
import matplotlib.pyplot as plt

from src.ai.algorithms.utils import *

##############################################################
# Settings
result_file_name = 'ppo_GAE_cartpole'
environment_name = 'CartPole-v0'  # 'Acrobot-v1'  # 'CartPole-v1'
learning_rate = 1e-3
epochs = 1000
batch_size = 1000
discount = 0.95
gae_lambda = 0.95
plot = True
policy_training_iterations = 10
value_training_iterations = 3
kl_target = 0.01
epsilon_ppo = 0.2
##############################################################
# Global variables:

environment = gym.make(environment_name)
discrete = isinstance(environment.action_space, gym.spaces.Discrete)

observation_dimension = environment.observation_space.shape[0]
action_dimension = environment.action_space.n if discrete else environment.action_space.shape[0]

policy_network = nn.Sequential(
    nn.Linear(observation_dimension, 20), nn.ReLU(inplace=True),
    nn.Linear(20, 20), nn.ReLU(inplace=True),
    nn.Linear(20, 20), nn.ReLU(inplace=True),
    nn.Linear(20, action_dimension),
)
if not discrete:
    log_std = torch.nn.Parameter(torch.as_tensor(-0.5 * np.ones(action_dimension, dtype=np.float32)))

optimizer = Adam(policy_network.parameters(), lr=learning_rate)
value_network = nn.Sequential(
    nn.Linear(observation_dimension, 20), nn.ReLU(inplace=True),
    nn.Linear(20, 20), nn.ReLU(inplace=True),
    nn.Linear(20, 20), nn.ReLU(inplace=True),
    nn.Linear(20, 1)
)
value_optimizer = Adam(value_network.parameters(), lr=learning_rate)


def policy_forward(observation):
    logits = policy_network(observation)
    if discrete:
        return Categorical(logits=logits)
    else:
        return Normal(logits, torch.exp(log_std))


def evaluate_policy(observation):
    output = policy_forward(torch.as_tensor(observation, dtype=torch.float32)).sample()
    if discrete:
        return output.item()
    else:
        return output


def train_one_epoch():
    batch_log_probabilities = []
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
        action = evaluate_policy(observation)

        batch_log_probabilities.append(
            policy_forward(torch.as_tensor(observation, dtype=torch.float32)).log_prob(torch.as_tensor(action))
        )
        observation, reward, done, info = environment.step(action)
        episode_rewards.append(reward)
        batch_actions.append(action)
        batch_done.append(done)
        batch_rewards.append(reward)
        if done:
            episode_return = sum(episode_rewards)
            episode_length = len(episode_rewards)
            epoch_returns.append(episode_return)
            episode_lengths.append(episode_length)
            observation = environment.reset()
            done = False
            episode_rewards = []

            if len(batch_observations) > batch_size:
                break

    # calculate batch weights
    values = value_network(torch.as_tensor(batch_observations, dtype=torch.float32))
    batch_advantages = generalized_advantage_estimate(batch_rewards=batch_rewards,
                                                      batch_done=batch_done,
                                                      batch_values=[v for v in values],
                                                      discount=discount,
                                                      gae_lambda=gae_lambda)

    # optimize policy with policy gradient step

    def compute_loss(obs, act, adv, logp):
        policy_forward(observation=torch.as_tensor(obs, dtype=torch.float32))
        new_log_probabilities = policy_forward(observation=torch.as_tensor(obs, dtype=torch.float32)).log_prob(act)
        ratio = torch.exp(new_log_probabilities - logp)
        batch_loss = -torch.min(ratio * adv, ratio.clamp(1-epsilon_ppo, 1+epsilon_ppo) * adv).mean()

        kl_approximation = (logp - new_log_probabilities).mean().item()
        return batch_loss, kl_approximation

    for policy_train_it in range(policy_training_iterations):
        optimizer.zero_grad()
        batch_loss, kl_approximation = compute_loss(obs=torch.as_tensor(batch_observations, dtype=torch.float32),
                                                    act=torch.as_tensor(batch_actions, dtype=torch.float32),
                                                    adv=torch.as_tensor(batch_advantages, dtype=torch.float32),
                                                    logp=torch.as_tensor(batch_log_probabilities, dtype=torch.float32))
        if kl_approximation > 1.5*kl_target:
            break
        batch_loss.backward()
        optimizer.step()

    # optimize value estimator
    def compute_value_loss(obs, trgt):
        return ((value_network(obs) - trgt) ** 2).mean()

    for value_train_it in range(value_training_iterations):
        value_optimizer.zero_grad()
        value_loss = compute_value_loss(obs=torch.as_tensor(batch_observations, dtype=torch.float32),
                                        trgt=torch.as_tensor(batch_advantages, dtype=torch.float32))
        value_loss.backward()
        value_optimizer.step()

    return batch_loss, epoch_returns, episode_lengths

############################################
# Loop over epochs and keep returns


start_time = time.time()

avg_returns = []
min_returns = []
max_returns = []

best_avg_return = -10000
result_directory = f'{os.environ["HOME"]}/code/imitation-learning-codebase/src/ai/algorithms/standalone_scripts/' \
                   f'results/ppo'
os.makedirs(result_directory, exist_ok=True)

for i in range(epochs):
    batch_loss, batch_returns, batch_lengths = train_one_epoch()
    avg_returns.append(np.mean(batch_returns))
    min_returns.append(min(batch_returns))
    max_returns.append(max(batch_returns))
    print(f'epoch: {i} \t loss: {batch_loss:0.3f} \t '
          f'returns: {np.mean(batch_returns): 0.3f} [{np.std(batch_returns): 0.3f}] \t'
          f'lengths: {np.mean(batch_lengths): 0.3f} [{np.std(batch_lengths): 0.3f}]')
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
