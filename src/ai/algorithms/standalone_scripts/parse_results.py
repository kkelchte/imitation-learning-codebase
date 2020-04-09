import gym
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
from torch.distributions.categorical import Categorical


def plot_results():
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    result_directory = f'{os.environ["HOME"]}/code/imitation-learning-codebase/src/ai/algorithms/standalone_scripts/results'
    result_files = os.listdir(result_directory)
    result_files = ['vanilla.npy']
    for index, result_file in enumerate(result_files):
        results = np.load(os.path.join(result_directory, result_file), 'r')
        plt.fill_between(range(len(results[0])), results[1], results[2], color=colors[index], alpha=0.5)
        plt.plot(results[0], color=colors[index], label=result_file.split('.')[0].replace('_', ' '))
    plt.legend()
    plt.show()


def evaluate_network(path_to_traced_model: str, environment: str, render: bool = True):
    with open(path_to_traced_model, 'rb') as f:
        model = torch.jit.load(path_to_traced_model)

    environment = gym.make(environment)
    observation = environment.reset()
    episode_rewards = []

    done = False
    while not done:
        outputs = model.forward(torch.as_tensor(observation, dtype=torch.float32))
        if isinstance(environment.action_space, gym.spaces.Discrete):
            action = Categorical(logits=outputs).sample().item()
        else:
            action = environment.action_space.high[0] * outputs.detach()
        observation, reward, done, info = environment.step(action)
        if render:
            environment.render()
        episode_rewards.append(reward)
    environment.close()


if __name__ == '__main__':
    evaluate_network(f'{os.environ["HOME"]}/code/imitation-learning-codebase/src/ai/algorithms'
                     f'/standalone_scripts/results/ddpg/policy.traced', 'Pendulum-v0')

