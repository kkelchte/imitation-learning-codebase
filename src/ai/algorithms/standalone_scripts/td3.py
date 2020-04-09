#!/usr/bin/python3.7
import os
import time
from copy import deepcopy
import numpy as np

from tqdm import tqdm
import gym
from gym.wrappers.time_limit import TimeLimit
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.optim import Adam
import matplotlib.pyplot as plt

from src.ai.algorithms.standalone_scripts.utils import *

##############################################################
# Settings
from src.sim.common.noise import OUNoise

result_file_name = 'td3_pendulum'
environment_name = 'Pendulum-v0'
actor_learning_rate = 0.0001
critic_learning_rate = 0.0001
epochs = 1000
batch_size = 256
replay_buffer_prefill = 20000
replay_buffer_size = 1000000
plot = True
optimize_every_n_steps = 100
training_iterations = 100
evaluation_iterations = 10
discount = 0.99
polyak_average = 0.995  # the closer to one, the slower the target network updates
prioritized_replay = False
seed = 123
update_policy_every_n_q_updates = 2
target_action_smoothing_sigma = 0.1
target_action_smoothing_boundary = 0.2
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
q_first = nn.Sequential(
    nn.Linear(observation_dimension + action_dimension, 15), nn.ReLU(inplace=True),
    nn.Linear(15, 15), nn.ReLU(inplace=True),
    nn.Linear(15, 15), nn.ReLU(inplace=True),
    nn.Linear(15, 1)
)
q_second = nn.Sequential(
    nn.Linear(observation_dimension + action_dimension, 15), nn.ReLU(inplace=True),
    nn.Linear(15, 15), nn.ReLU(inplace=True),
    nn.Linear(15, 15), nn.ReLU(inplace=True),
    nn.Linear(15, 1)
)
target_policy = deepcopy(policy)
target_q_first = deepcopy(q_first)
target_q_second = deepcopy(q_second)

for network in [target_policy, target_q_first, target_q_second]:
    for p in network.parameters():
        p.requires_grad = False

policy_optimizer = Adam(policy.parameters(), lr=actor_learning_rate)
q_first_optimizer = Adam(q_first.parameters(), lr=critic_learning_rate)
q_second_optimizer = Adam(q_second.parameters(), lr=critic_learning_rate)

epsilon = 1
exploration = torch.distributions.Uniform(low=action_lower_bound,
                                          high=action_higher_bound)
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


def evaluate_target_q_twin(observation):
    if not isinstance(observation, torch.Tensor):
        observation = torch.as_tensor(observation, dtype=torch.float32)
    action = action_higher_bound * target_policy.forward(observation)
    # add target policy smoothing by adding gaussian clipped noise
    action += torch.distributions.normal.Normal(0, target_action_smoothing_sigma).sample()\
        .clamp(target_action_smoothing_boundary)
    action = action.clamp(action_lower_bound, action_higher_bound)
    stacked_tensors = torch.cat((observation, action), dim=-1)
    # select minimum from twin q functions for target
    return min(target_q_first.forward(stacked_tensors), target_q_second.forward(stacked_tensors))


def evaluate_q(observation, q):
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
        with torch.no_grad():
            action = evaluate_policy(observation, evaluate=False)
        next_observation, reward, done, info = environment.step(action.detach().numpy())
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
    policy_objectives = []
    for n in range(training_iterations):
        batch_observations, batch_rewards, batch_actions, batch_next_observations, batch_done, batch_indices = \
            sample_batch()
        # calculate q target values with bellman update
        with torch.no_grad():
            q_targets = [
                batch_rewards[index] + discount * (1-batch_done[index]) *
            evaluate_target_q_twin(batch_next_observations[index]).detach() for index in range(batch_size)
        ]
        # update q-function with one step gradient descent
        for p in policy.parameters():
            p.requires_grad = False
        losses = []
        for optimizer, q in [(q_first_optimizer, q_first), (q_second_optimizer, q_second)]:
            optimizer.zero_grad()
            q_loss = ((evaluate_q(batch_observations, q)
                       - torch.as_tensor(q_targets, dtype=torch.float32))**2)
            losses.append(q_loss.detach().tolist())
            q_loss_avg = q_loss.mean()
            q_loss_avg.backward()
            optimizer.step()
            q_losses.append(q_loss_avg.detach().item())
        if prioritized_replay:
            store_loss_for_prioritized_experience_replay([losses[0][i]+losses[1][i] for i in range(len(losses[0]))],
                                                     batch_indices)
        for p in policy.parameters():
            p.requires_grad = True
        

        # update policy function with one step gradient ascent
        if n % update_policy_every_n_q_updates == 0:
            policy_optimizer.zero_grad()
            for p in q_first.parameters():
                p.requires_grad = False
            policy_objective = -evaluate_q(batch_observations, q_first).mean()  # invert for ascent
            policy_objective.backward()
            policy_optimizer.step()
            policy_objectives.append(policy_objective.item())
            for p in q_first.parameters():
                p.requires_grad = True
            for network, target_network in [(q_first, target_q_first),
                               (q_second, target_q_second),
                               (policy, target_policy)]:
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
            observation, reward, done, info = environment.step(action.detach().numpy())
            if n == 0 and render:
                environment.render()
            episode_rewards.append(reward)
        epoch_returns.append(sum(episode_rewards))
    return epoch_returns, q_losses, policy_objectives

############################################
# Loop over epochs and keep returns


start_time = time.time()

avg_returns = []
min_returns = []
max_returns = []

best_avg_return = -10000
result_directory = f'{os.environ["HOME"]}/code/imitation-learning-codebase/src/ai/algorithms/standalone_scripts/' \
                   f'results/td3'
os.makedirs(result_directory, exist_ok=True)

for epoch in range(epochs):
    evaluation_returns, losses, objectives = train_one_epoch(epoch=epoch, render=(epoch % 20 == 0) and epoch != 0)
    avg_returns.append(np.mean(evaluation_returns))
    min_returns.append(min(evaluation_returns))
    max_returns.append(max(evaluation_returns))
    print(f'epoch: {epoch} \t returns: {np.mean(evaluation_returns): 0.3f} [{np.std(evaluation_returns): 0.3f}] \t'
          f'critic loss: {np.mean(losses): 0.3f} [{np.std(losses): 0.3f}], '
          f'policy objective: {np.mean(objectives): 0.3f}[{np.std(objectives): 0.3f}]')
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
environment.close()
