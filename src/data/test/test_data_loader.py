import shutil
import unittest
import os
from copy import deepcopy

import numpy as np

from src.data.data_loader import DataLoader, DataLoaderConfig
from src.core.utils import get_filename_without_extension
from src.data.data_saver import DataSaverConfig, DataSaver
from src.data.test.common_utils import generate_dummy_dataset
from src.data.utils import arrange_run_according_timestamps, calculate_weights


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
        self.info = generate_dummy_dataset(self.data_saver, num_runs=20)

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

    def test_calculate_weights_for_data_balancing_uniform(self):
        # uniform distributed case:
        data = np.random.uniform(0, 1, 1000)
        weights = calculate_weights(data)
        print(max(weights) - min(weights))
        self.assertTrue(max(weights) - min(weights) < 0.3)

    def test_calculate_weights_for_data_balancing_normal(self):
        # normal distributed case:
        data = np.random.normal(0, 1, 1000)
        weights = calculate_weights(data)
        print(max(weights) - min(weights))
        self.assertTrue(0.5 < max(weights) - min(weights) < 1.5)

    def test_calculate_weights_for_data_balancing_discrete(self):
        # discrete distributions:
        data = [0] * 30 + [1] * 70
        weights = calculate_weights(data)
        self.assertEqual(weights[0], weights[29])
        self.assertNotEqual(weights[29], weights[30])
        self.assertTrue(weights[0] - 0.777 < 0.1)
        self.assertTrue(weights[30] - 0.333 < 0.1)

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
        for batch in data_loader.sample_shuffled_batch(batch_size=20):
            action_variances_with_balancing.append(
                [np.std([a[dim] for a in batch.actions]) for dim in range(batch.actions[0].size()[0])]
            )
        action_variances_with_balancing = np.asarray(action_variances_with_balancing)

        config = DataLoaderConfig().create(config_dict={
            'data_directories': self.info['episode_directories'],
            'output_path': self.output_dir,
            'balance_over_actions': False
        })
        data_loader = DataLoader(config=config)
        data_loader.load_dataset(arrange_according_to_timestamp=False)
        action_variances_without_balancing = []
        for batch in data_loader.sample_shuffled_batch(batch_size=20):
            action_variances_without_balancing.append(
                [np.std([a[dim] for a in batch.actions]) for dim in range(batch.actions[0].size()[0])]
            )

        action_variances_without_balancing = np.asarray(action_variances_without_balancing)
        for dim in range(action_variances_with_balancing.shape[-1]):
            self.assertTrue(np.mean(action_variances_with_balancing[:, dim]) >=
                            np.mean(action_variances_without_balancing[:, dim]))

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
