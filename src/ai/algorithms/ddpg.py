#!/usr/bin/python3.8
import os
import time
from copy import deepcopy
import numpy as np

import gym
from gym.wrappers.time_limit import TimeLimit
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from src.ai.algorithms.utils import *

##############################################################
# Settings

result_file_name = 'ddpg_pendulum'
environment_name = 'Pendulum-v0'  # 'MountainCarContinuous-v0'
actor_learning_rate = 0.00001
critic_learning_rate = 0.00001
epochs = 2000
batch_size = 256
replay_buffer_prefill = 2000
replay_buffer_size = 1000000
plot = True
optimize_every_n_steps = 200
training_iterations = 200
evaluation_iterations = 10
discount = 0.99
polyak_average = 0.995  # the closer to one, the slower the target network updates
prioritized_replay = False
seed = 123
##############################################################
np.random.seed(seed)
torch.manual_seed(seed)

# Global variables:
environment = TimeLimit(gym.make(environment_name), max_episode_steps=200)
environment.seed(seed)
assert isinstance(environment.action_space, gym.spaces.Box)

observation_dimension = environment.observation_space.shape[0]
action_dimension = environment.action_space.shape[0]
action_lower_bound = environment.action_space.low[0]
action_higher_bound = environment.action_space.high[0]

policy = nn.Sequential(
    nn.Linear(observation_dimension, 15), nn.ReLU(inplace=True),
    nn.Linear(15, 15), nn.ReLU(inplace=True),
    nn.Linear(15, action_dimension), nn.Tanh()
)


q = nn.Sequential(
    nn.Linear(observation_dimension+action_dimension, 15), nn.ReLU(inplace=True),
    nn.Linear(15, 15), nn.ReLU(inplace=True),
    nn.Linear(15, 15), nn.ReLU(inplace=True),
    nn.Linear(15, 1)
)

target_policy = deepcopy(policy)
target_q = deepcopy(q)

for network in [target_policy, target_q]:
    for p in network.parameters():
        p.requires_grad = False

policy_optimizer = Adam(policy.parameters(), lr=actor_learning_rate)
q_optimizer = Adam(q.parameters(), lr=critic_learning_rate)
epsilon = 1
exploration = torch.distributions.Normal(loc=action_lower_bound + (action_higher_bound - action_lower_bound)/2,
                                         scale=(action_higher_bound - action_lower_bound)/4)
# exploration = OUNoise()
replay_buffer = {
    'state': [],
    'action': [],
    'reward': [],
    'next_state': [],
    'terminal': [],
    'probability': [],
}


def evaluate_policy(observation, evaluate: bool = True):
    if not isinstance(observation, torch.Tensor):
        observation = torch.as_tensor(observation, dtype=torch.float32)
    action = action_higher_bound * policy.forward(observation)

    if not evaluate:
        action += epsilon * torch.as_tensor(exploration.sample(), dtype=torch.float32)
    return torch.clamp(action, action_lower_bound, action_higher_bound)


def evaluate_target_q(observation):
    if not isinstance(observation, torch.Tensor):
        observation = torch.as_tensor(observation, dtype=torch.float32)
    action = action_higher_bound * target_policy.forward(observation)
    stacked_tensors = torch.cat((observation, action), dim=-1)
    return target_q.forward(stacked_tensors).squeeze()


def evaluate_q(observation):
    if not isinstance(observation, torch.Tensor):
        observation = torch.as_tensor(observation, dtype=torch.float32)
    action = action_higher_bound * policy.forward(observation)
    stacked_tensors = torch.cat((observation, action), dim=-1)
    return q.forward(stacked_tensors).squeeze()


def save_experience(state, action, reward, next_state, terminal):
    if len(replay_buffer['state']) == replay_buffer_size:
        for k in replay_buffer.keys():
            replay_buffer[k].pop(0)
    replay_buffer['state'].append(state)
    replay_buffer['action'].append(action)
    replay_buffer['reward'].append(reward)
    replay_buffer['next_state'].append(next_state)
    replay_buffer['terminal'].append(terminal)
    replay_buffer['probability'].append(300)


def normalize_probabilities(logits: list) -> list:
    return [p/sum(logits) for p in logits]


def sample_batch():
    batch_indices = np.random.choice(len(replay_buffer['state']), batch_size, replace=False,
                                     p=normalize_probabilities(replay_buffer['probability']) if prioritized_replay
                                     else None)
    batch_observations = [replay_buffer['state'][index] for index in batch_indices]
    batch_rewards = [replay_buffer['reward'][index] for index in batch_indices]
    batch_actions = [replay_buffer['action'][index] for index in batch_indices]
    batch_next_observations = [replay_buffer['next_state'][index] for index in batch_indices]
    batch_done = [replay_buffer['terminal'][index] for index in batch_indices]
    return batch_observations, batch_rewards, batch_actions, batch_next_observations, batch_done, batch_indices


def store_loss_for_prioritized_experience_replay(q_loss, batch_indices):
    assert len(q_loss) == len(batch_indices)
    for loss, index in zip(q_loss, batch_indices):
        replay_buffer['probability'][index] = max(0, loss)


def update_epsilon():
    global epsilon
    epsilon -= (1 - 0.1) / epochs


def train_one_epoch(epoch: int, render: bool = False):
    update_epsilon()
    observation = environment.reset()
    done = False
    step_count = 0
    while True:
        action = evaluate_policy(observation, evaluate=False).detach().numpy()
        next_observation, reward, done, info = environment.step(action)
        step_count += 1
        save_experience(state=observation,
                        action=action,
                        reward=reward,
                        next_state=next_observation,
                        terminal=done)
        observation = deepcopy(next_observation)

        if done:
            observation = environment.reset()
            done = False
            if epoch == 0:
                if step_count >= replay_buffer_prefill:
                    break
            else:
                if step_count >= optimize_every_n_steps:
                    break

    q_losses = []
    q_target_means = []
    q_target_stds = []
    q_estimate_mean = []
    q_estimate_std = []
    policy_objectives = []
    for n in range(training_iterations):
        batch_observations, batch_rewards, batch_actions, batch_next_observations, batch_done, batch_indices = \
            sample_batch()
        # calculate q target values with bellman update
        with torch.no_grad():
            q_targets = [
                batch_rewards[index] + discount * (1 - batch_done[index])
                * evaluate_target_q(batch_next_observations[index]).detach() for index in range(batch_size)
            ]
        q_target_means.append(np.mean(q_targets))
        q_target_stds.append(np.std(q_targets))

        # update q-function with one step gradient descent
        for p in policy.parameters():
            p.requires_grad = False
        q_optimizer.zero_grad()
        q_estimates = evaluate_q(batch_observations)
        q_estimate_mean.append(np.mean(q_estimates.detach().numpy()))
        q_estimate_std.append(np.std(q_estimates.detach().numpy()))
        q_loss = ((q_estimates - torch.as_tensor(q_targets, dtype=torch.float32)) ** 2)
        if prioritized_replay:
            store_loss_for_prioritized_experience_replay(q_loss.detach().numpy(), batch_indices)
        q_loss_avg = q_loss.mean()
        q_loss_avg.backward()
        q_optimizer.step()
        for p in policy.parameters():
            p.requires_grad = True
        q_losses.append(q_loss_avg.detach().item())

        # update policy function with one step gradient ascent
        for p in q.parameters():
            p.requires_grad = False
        policy_optimizer.zero_grad()
        policy_objective = -evaluate_q(batch_observations).mean()  # invert for ascent
        policy_objective.backward()
        policy_optimizer.step()
        for p in q.parameters():
            p.requires_grad = True
        policy_objectives.append(policy_objective.detach().item())

        # update target policy and target q function with polyak average
        for network, target_network in [(policy, target_policy),
                                        (q, target_q)]:
            with torch.no_grad():
                for p, target_p in zip(network.parameters(), target_network.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    target_p.data.mul_(polyak_average)
                    target_p.data.add_((1 - polyak_average) * p.data)

    # evaluate policy without exploration noise
    epoch_returns = []
    for n in range(evaluation_iterations):
        observation = environment.reset()
        episode_rewards = []
        done = False
        while not done:
            action = evaluate_policy(observation, evaluate=True).detach().numpy()
            observation, reward, done, info = environment.step(action)
            if n == 0 and render:
                environment.render()
            episode_rewards.append(reward)
        epoch_returns.append(sum(episode_rewards))
    return epoch_returns, q_losses, policy_objectives, q_target_means, q_target_stds, q_estimate_mean, q_estimate_std

############################################
# Loop over epochs and keep returns


start_time = time.time()

avg_returns = []
min_returns = []
max_returns = []

best_avg_return = -10000
result_directory = f'{os.environ["HOME"]}/code/imitation-learning-codebase/src/ai/algorithms/' \
                   f'results/ddpg'
os.makedirs(result_directory, exist_ok=True)

linear_regressor = LinearRegression()
for epoch in range(epochs):
    evaluation_returns, losses, objectives, \
    q_target_means, q_target_stds, \
    q_estimate_mean, q_estimate_std = train_one_epoch(epoch=epoch, render=(epoch % 20 == 0) and epoch != 0)
    loss_coef = linear_regressor.fit(np.arange(len(losses)).reshape((-1, 1)),
                                     np.asarray(losses).reshape((-1, 1))).coef_.item()
    objectives_coef = linear_regressor.fit(np.arange(len(objectives)).reshape((-1, 1)),
                                           np.asarray(objectives).reshape((-1, 1))).coef_.item()

    avg_returns.append(np.mean(evaluation_returns))
    min_returns.append(min(evaluation_returns))
    max_returns.append(max(evaluation_returns))
    print(f'epoch: {epoch} returns: {np.mean(evaluation_returns): 0.3f} [{np.std(evaluation_returns): 0.3f}] '
          f'critic: {np.mean(losses): 0.3f} [std:{np.std(losses): 0.3f}, d: {loss_coef: 0.3f}], '
          f'policy: {np.mean(objectives): 0.3f} [std:{np.std(objectives): 0.3f}, d: {objectives_coef: 0.3f}], '
          f'q_target_means: {np.mean(q_target_means): 0.3f} [std:{np.std(q_target_means): 0.1f}], '
          f'q_target_stds: {np.mean(q_target_stds): 0.3f}, '
          f'q_estimate_mean: {np.mean(q_estimate_mean): 0.3f} [std:{np.std(q_estimate_mean): 0.1f}], '
          f'q_estimate_std: {np.mean(q_estimate_std): 0.3f} ')
#          f'replay buffer reward: {np.mean(replay_buffer["reward"])} '
#          f'[{np.min(replay_buffer["reward"]): 0.3f}: {np.max(replay_buffer["reward"]): 0.3f}]')
    if (epoch % 20 == 10 or epoch == epochs - 1) and plot:
        plt.fill_between(range(len(avg_returns)), min_returns, max_returns, color='blue', alpha=0.5)
        plt.plot(avg_returns, color='b')
        plt.show()

    if np.mean(evaluation_returns) > best_avg_return:
        store_checkpoint(policy, result_directory + '/policy')
        best_avg_return = np.mean(evaluation_returns)

print(f'total duration : {time.time()-start_time}s')
results = np.asarray([avg_returns, min_returns, max_returns])
np.save(os.path.join(result_directory, result_file_name + '.npy'), results)
