import os
import shutil
import unittest
from copy import deepcopy

import numpy as np
import torch

from src.ai.base_net import InitializationType, ArchitectureConfig
from src.ai.evaluator import Evaluator, EvaluatorConfig
from src.ai.trainer import TrainerConfig, Trainer
from src.ai.utils import generate_random_dataset_in_raw_data, get_returns, get_reward_to_go, \
    get_generalized_advantage_estimate
from src.core.data_types import Dataset, Experience
from src.core.utils import get_to_root_dir, get_filename_without_extension
from src.ai.architectures import *  # Do not remove
from src.data.data_loader import DataLoaderConfig, DataLoader


class TrainerTest(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        self.batch = Dataset()
        self.durations = [10, 1, 5]
        self.step_reward = torch.as_tensor(1)
        self.end_reward = torch.as_tensor(10)
        for episode in range(3):
            for experience in range(self.durations[episode] - 1):
                self.batch.append(Experience(
                    observation=torch.as_tensor(5),
                    action=torch.as_tensor(5),
                    reward=self.step_reward,
                    done=torch.as_tensor(0)
                ))
            self.batch.append(Experience(
                observation=torch.as_tensor(5),
                action=torch.as_tensor(5),
                reward=self.end_reward,
                done=torch.as_tensor(2)
            ))

    def test_get_returns_on_dataset(self):
        returns = get_returns(self.batch)
        targets = [
            self.end_reward + (duration - 1) * self.step_reward for duration in self.durations for _ in range(duration)
        ]
        for r_e, r_t in zip(returns, targets):
            self.assertEqual(r_e, r_t)

    def test_get_reward_to_go(self):
        returns = get_reward_to_go(self.batch)
        targets = reversed([self.end_reward + t * self.step_reward
                            for duration in reversed(self.durations)
                            for t in range(duration)])

        for r_e, r_t in zip(returns, targets):
            self.assertEqual(r_e, r_t)

    def test_generalized_advantage_estimate(self):
        # with gae_lambda == 1 and no value --> same as reward-to-go
        rtg_returns = get_generalized_advantage_estimate(
            batch_rewards=self.batch.rewards,
            batch_done=self.batch.done,
            batch_values=[torch.as_tensor(0.)] * len(self.batch),
            discount=1,
            gae_lambda=1
        )
        for r_e, r_t in zip(rtg_returns, get_reward_to_go(self.batch)):
            self.assertEqual(r_e, r_t)

        one_step_returns = get_generalized_advantage_estimate(
            batch_rewards=self.batch.rewards,
            batch_done=self.batch.done,
            batch_values=[torch.as_tensor(0.)] * len(self.batch),
            discount=1,
            gae_lambda=0
        )
        targets = [self.step_reward if d == 0 else self.end_reward for d in self.batch.done]
        for r_e, r_t in zip(one_step_returns, targets):
            self.assertEqual(r_e, r_t)

        gae_returns = get_generalized_advantage_estimate(
            batch_rewards=self.batch.rewards,
            batch_done=self.batch.done,
            batch_values=[torch.as_tensor(0.)] * len(self.batch),
            discount=0.99,
            gae_lambda=0.99
        )
        for t in range(len(self.batch)):
            self.assertGreaterEqual(gae_returns[t], one_step_returns[t])
            self.assertLessEqual(gae_returns[t], rtg_returns[t])

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
