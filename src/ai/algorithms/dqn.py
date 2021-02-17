#!/usr/bin/python3.8
import os
import time
from copy import deepcopy
import numpy as np

from tqdm import tqdm
import gym
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import matplotlib.pyplot as plt

from src.ai.algorithms.utils import *

##############################################################
# Settings

result_file_name = 'dqn_cartpole'
environment_name = 'CartPole-v0'
learning_rate = 0.0005
epochs = 100
batch_size = 64
replay_buffer_prefill = 1000
replay_buffer_size = 100000
plot = True
training_iterations = 100
optimize_every_n_steps = 100
evaluation_iterations = 10
discount = 0.99
polyak_average = 0.  # originally should be strong update
prioritized_replay = False
seed = 123
##############################################################
np.random.seed(seed)
torch.manual_seed(seed)

# Global variables:
environment = gym.make(environment_name)
environment.seed(seed)

observation_dimension = environment.observation_space.shape[0]
action_dimension = environment.action_space.n

q = nn.Sequential(
    nn.Linear(observation_dimension, 20), nn.ReLU(inplace=True),
    nn.Linear(20, 20), nn.ReLU(inplace=True),
    nn.Linear(20, 20), nn.ReLU(inplace=True),
    nn.Linear(20, action_dimension)
)
target_q = deepcopy(q)

for p in target_q.parameters():
    p.requires_grad = False

q_optimizer = Adam([p for p in q.parameters()], lr=learning_rate)
epsilon = 1
exploration = torch.distributions.Categorical(logits=torch.as_tensor((0.5, 0.5)))
#exploration = OUNoise(mean=0.5, std=0.5, pullback=0.15)
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
    action = torch.argmax(q.forward(observation), dim=-1).item()
    if not evaluate and torch.distributions.binomial.Binomial(probs=epsilon).sample():
        action = exploration.sample().item()
    return action


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
        action = evaluate_policy(observation, evaluate=False)
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
    for n in range(training_iterations):
        batch_observations, batch_rewards, batch_actions, batch_next_observations, batch_done, batch_indices = \
            sample_batch()
        # calculate q target values with bellman update
        with torch.no_grad():
            q_targets = [
                batch_rewards[index] + discount * (1 - batch_done[index]) *
                torch.max(q.forward(torch.as_tensor(batch_next_observations[index], dtype=torch.float32))).detach()
                for index in range(batch_size)
            ]
        # update q-function with one step gradient descent
        q_optimizer.zero_grad()

        def to_one_hot(x: torch.Tensor, dimension: int = action_dimension):
            y = torch.zeros((x.shape[0], dimension))
            y[torch.arange(0, x.shape[0], dtype=torch.long), x.type(torch.long)] = 1
            return y.type(torch.bool)

        def evaluate_q(obs, act):
            if not isinstance(obs, torch.Tensor):
                obs = torch.as_tensor(obs, dtype=torch.float32)
            if not isinstance(act, torch.Tensor):
                act = torch.as_tensor(act, dtype=torch.float32)
            return torch.masked_select(q.forward(obs), mask=to_one_hot(act, dimension=action_dimension))

        q_loss = ((evaluate_q(batch_observations, batch_actions)
                   - torch.as_tensor(q_targets, dtype=torch.float32)) ** 2)
        if prioritized_replay:
            store_loss_for_prioritized_experience_replay(q_loss.detach().numpy(), batch_indices)
        q_loss_avg = q_loss.mean()
        q_loss_avg.backward()
        q_optimizer.step()

        # update target policy and target q function with polyak average
        for network, target_network in [(q, target_q)]:
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
            action = evaluate_policy(observation, evaluate=True)
            observation, reward, done, info = environment.step(action)
            if n == 0 and render:
                environment.render()
            episode_rewards.append(reward)
        epoch_returns.append(sum(episode_rewards))
    return epoch_returns

############################################
# Loop over epochs and keep returns


start_time = time.time()

avg_returns = []
min_returns = []
max_returns = []

best_avg_return = -10000
result_directory = f'{os.environ["HOME"]}/code/imitation-learning-codebase/src/ai/algorithms/' \
                   f'results/dqn'
os.makedirs(result_directory, exist_ok=True)

for epoch in tqdm(range(epochs)):
    batch_returns = train_one_epoch(epoch=epoch, render=(epoch % 20 == 0) and epoch != 0)
    avg_returns.append(np.mean(batch_returns))
    min_returns.append(min(batch_returns))
    max_returns.append(max(batch_returns))
    print(f'epoch: {epoch} \t returns: {np.mean(batch_returns): 0.3f} [{np.std(batch_returns): 0.3f}] \t'
          f'replay buffer reward: {np.mean(replay_buffer["reward"])} '
          f'[{np.min(replay_buffer["reward"]): 0.3f}: {np.max(replay_buffer["reward"]): 0.3f}]')
    if (epoch % 20 == 10 or epoch == epochs - 1) and plot:
        plt.fill_between(range(len(avg_returns)), min_returns, max_returns, color='blue', alpha=0.5)
        plt.plot(avg_returns, color='b')
        plt.show()
    if np.mean(batch_returns) > best_avg_return:
        store_checkpoint(q, result_directory + '/policy')
        best_avg_return = np.mean(batch_returns)

print(f'total duration : {time.time()-start_time}s')
results = np.asarray([avg_returns, min_returns, max_returns])
np.save(os.path.join(result_directory, result_file_name + '.npy'), results)
