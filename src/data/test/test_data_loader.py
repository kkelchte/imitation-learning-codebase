import shutil
import unittest
import os
from copy import deepcopy
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.data.data_types import Dataset
from src.data.dataset_loader import DataLoader, DataLoaderConfig
from src.core.utils import get_filename_without_extension
from src.data.dataset_saver import DataSaverConfig, DataSaver
from src.data.test.common_utils import generate_dummy_dataset
from src.data.utils import arrange_run_according_timestamps, calculate_weights, balance_weights_over_actions


class TestDataLoader(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        config_dict = {
            'output_path': self.output_dir,
            'store_hdf5': True
        }
        config = DataSaverConfig().create(config_dict=config_dict)
        self.data_saver = DataSaver(config=config)
        self.info = generate_dummy_dataset(self.data_saver, num_runs=2)

    def test_data_loading(self):
        config_dict = {
            'data_directories': self.info['episode_directories'],
            'output_path': self.output_dir,
        }
        config = DataLoaderConfig().create(config_dict=config_dict)
        data_loader = DataLoader(config=config)
        data_loader.load_dataset(arrange_according_to_timestamp=False)

        # assert nothing is empty
        for k in ['observations', 'actions', 'rewards', 'done']:
            data = eval(f'data_loader.get_dataset().{k}')
            self.assertTrue(len(data) > 0)
            self.assertTrue(sum(data[0].shape) > 0)
            print(f'loaded {len(data)} data points of shape {data[0].shape}')

    def test_arrange_run_according_timestamps(self):
        # Normal usage where all should be kept
        num = 3
        run = {
            'a': list(range(num)),
            'b': list(range(num))
        }
        backup_run = deepcopy(run)
        time_stamps = {
            'a': list(range(num)),
            'b': list(range(num))
        }
        result = arrange_run_according_timestamps(run=run, time_stamps=time_stamps)
        self.assertEqual(result, backup_run)
        # Rearrange required
        num = 3
        run = {
            'a': list(range(num)),
            'b': list(range(num))
        }
        backup_run = deepcopy(run)
        time_stamps = {
            'a': list(range(num)),
            'b': [i+1 for i in range(num)]
        }
        result = arrange_run_according_timestamps(run=run, time_stamps=time_stamps)
        self.assertEqual(result['a'], backup_run['a'][1:])
        self.assertEqual(result['b'], backup_run['b'][:-1])

    def test_data_loader_with_relative_paths(self):
        config_dict = {
            'data_directories': ['raw_data/' + os.path.basename(p) for p in self.info['episode_directories']],
            'output_path': self.output_dir,
        }
        config = DataLoaderConfig().create(config_dict=config_dict)
        data_loader = DataLoader(config=config)
        data_loader.load_dataset()

        config = DataLoaderConfig().create(config_dict=config_dict)
        for d in config.data_directories:
            self.assertTrue(os.path.isdir(d))

    def test_sample_batch(self):
        batch_size = 3
        max_num_batches = 2
        config_dict = {
            'data_directories': self.info['episode_directories'],
            'output_path': self.output_dir,
        }
        config = DataLoaderConfig().create(config_dict=config_dict)
        data_loader = DataLoader(config=config)
        data_loader.load_dataset(arrange_according_to_timestamp=False)
        index = 0
        for index, batch in enumerate(data_loader.sample_shuffled_batch(batch_size=batch_size,
                                                                        max_number_of_batches=max_num_batches)):
            self.assertEqual(len(batch), batch_size)
        self.assertEqual(index, max_num_batches - 1)

    def test_sample_batch(self):
        batch_size = 3
        max_num_batches = 5
        config_dict = {
            'data_directories': self.info['episode_directories'],
            'output_path': self.output_dir,
        }
        config = DataLoaderConfig().create(config_dict=config_dict)
        data_loader = DataLoader(config=config)
        data_loader.load_dataset(arrange_according_to_timestamp=False)
        index = 0
        for index, batch in enumerate(data_loader.sample_shuffled_batch(batch_size=batch_size,
                                                                        max_number_of_batches=max_num_batches)):
            self.assertEqual(len(batch), batch_size)
        self.assertEqual(index, max_num_batches - 1)

    def test_calculate_weights_for_data_balancing_uniform(self):
        # uniform distributed case:
        data = np.random.uniform(0, 1, 1000)
        weights = calculate_weights(data)
        print(max(weights) - min(weights))
        self.assertTrue(max(weights) - min(weights) < 0.001)

    def test_calculate_weights_for_data_balancing_normal(self):
        # normal distributed case:
        data = np.random.normal(0, 1, 1000)
        weights = calculate_weights(data)
        self.assertTrue(0.2 < max(weights) - min(weights) < 0.8)

    def test_calculate_weights_for_data_balancing_discrete(self):
        # discrete distributions:
        data = [0] * 30 + [1] * 70
        weights = calculate_weights(data)
        self.assertTrue(max(weights) - min(weights) < 10e-5)

    def test_data_balancing(self):
        # average action variance in batch with action balancing
        config = DataLoaderConfig().create(config_dict={
            'data_directories': self.info['episode_directories'],
            'output_path': self.output_dir,
            'balance_over_actions': True
        })
        data_loader = DataLoader(config=config)
        data_loader.load_dataset(arrange_according_to_timestamp=False)
        action_variances_with_balancing = []
        for batch in data_loader.sample_shuffled_batch(batch_size=10):
            action_variances_with_balancing.append(
                [np.var([a[dim] for a in batch.actions]) for dim in range(batch.actions[0].size()[0])]
            )

        config = DataLoaderConfig().create(config_dict={
            'data_directories': self.info['episode_directories'],
            'output_path': self.output_dir,
            'balance_over_actions': False
        })
        data_loader = DataLoader(config=config)
        data_loader.load_dataset(arrange_according_to_timestamp=False)
        action_variances_without_balancing = []
        for batch in data_loader.sample_shuffled_batch(batch_size=10):
            action_variances_without_balancing.append(
                [np.var([a[dim] for a in batch.actions]) for dim in range(batch.actions[0].size()[0])]
            )

        self.assertTrue(np.mean(action_variances_with_balancing) > np.mean(action_variances_without_balancing))

    # def test_data_loaders_data_balancing(self):
    #     config_dict = {
    #         'data_directories': self.info['episode_directories'],
    #         'output_path': self.output_dir,
    #         'inputs': self.info['inputs'],
    #         'outputs': self.info['outputs']
    #     }
    #     config = DataLoaderConfig().create(config_dict=config_dict)
    #     data_loader = DataLoader(config=config)
    #     data_loader.load_dataset()
    #
    #     # test calculate probabilities for run
    #     data = [float(d) for d in data_loader._dataset.data[0].outputs['expert'][:, 5]]
    #     probabilities = calculate_probabilities(data)
    #
    #     # by sampling new data with probabilities and asserting difference among histogram bins is relatively low
    #     clean_data = []
    #     for i in range(300):
    #         clean_data.append(np.random.choice(data, p=probabilities))
    #     y, x, _ = plt.hist(clean_data, bins=get_ideal_number_of_bins(data))
    #     relative_height_difference = (max(y) - min(y[y != 0])) / max(y)
    #     self.assertTrue(relative_height_difference < 0.3)
    #
    #     # normalize over all actions should not have impact as all other actions are the same:
    #     probabilities_all_dimensions = calculate_probabilites_per_run(data_loader._dataset.data[0])
    #     self.assertTrue(np.abs(min(probabilities_all_dimensions) - min(probabilities)) < 1e-6)
    #     self.assertTrue(np.abs(max(probabilities_all_dimensions) - max(probabilities)) < 1e-6)
    #
    #     # sampling a large batch should have a low relative height
    #     clean_data = []
    #     for batch in data_loader.sample_shuffled_batch(batch_size=100):
    #         clean_data.extend([float(t) for t in batch.outputs['expert'][:, 5]])
    #     y, x, _ = plt.hist(clean_data, bins=get_ideal_number_of_bins(clean_data))
    #     relative_height_difference = (max(y) - min(y[y != 0])) / max(y)
    #     self.assertTrue(relative_height_difference < 0.3)

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
