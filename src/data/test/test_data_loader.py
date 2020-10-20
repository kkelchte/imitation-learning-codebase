import shutil
import unittest
import os
from copy import deepcopy

import numpy as np
import torch

from src.data.data_loader import DataLoader, DataLoaderConfig
from src.core.utils import get_filename_without_extension, get_data_dir
from src.data.data_saver import DataSaverConfig, DataSaver
from src.data.test.common_utils import generate_dummy_dataset
from src.data.utils import arrange_run_according_timestamps, calculate_weights, select


class TestDataLoader(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{get_data_dir(os.environ["HOME"])}/test_dir/{get_filename_without_extension(__file__)}'
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        config_dict = {
            'output_path': self.output_dir,
            'store_hdf5': True
        }
        config = DataSaverConfig().create(config_dict=config_dict)
        self.data_saver = DataSaver(config=config)

    def test_data_loading(self):
        self.info = generate_dummy_dataset(self.data_saver, num_runs=20, input_size=(100, 100, 3), output_size=(3,),
                                           continuous=False)
        config_dict = {
            'data_directories': self.info['episode_directories'],
            'output_path': self.output_dir,
        }
        config = DataLoaderConfig().create(config_dict=config_dict)
        data_loader = DataLoader(config=config)
        data_loader.load_dataset()

        # assert nothing is empty
        for k in ['observations', 'actions', 'rewards', 'done']:
            data = eval(f'data_loader.get_dataset().{k}')
            self.assertTrue(len(data) > 0)
            self.assertTrue(sum(data[0].shape) > 0)

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

    def test_data_loader_from_raw_path_dirs(self):
        self.info = generate_dummy_dataset(self.data_saver, num_runs=20, input_size=(100, 100, 3), output_size=(3,),
                                           continuous=False)
        config_dict = {
            'data_directories': [self.output_dir],
            'output_path': self.output_dir,
        }
        config = DataLoaderConfig().create(config_dict=config_dict)
        data_loader = DataLoader(config=config)
        data_loader.load_dataset()

        config = DataLoaderConfig().create(config_dict=config_dict)
        for d in config.data_directories:
            self.assertTrue(os.path.isdir(d))

    def test_data_batch(self):
        self.info = generate_dummy_dataset(self.data_saver, num_runs=20, input_size=(100, 100, 3), output_size=(3,),
                                           continuous=False)
        config_dict = {
            'data_directories': self.info['episode_directories'],
            'output_path': self.output_dir,
            'random_seed': 1,
            'batch_size': 3
        }
        data_loader = DataLoader(config=DataLoaderConfig().create(config_dict=config_dict))
        data_loader.load_dataset()

        for batch in data_loader.get_data_batch():
            self.assertEqual(len(batch), config_dict['batch_size'])
            break

    def test_sample_batch(self):
        self.info = generate_dummy_dataset(self.data_saver, num_runs=20, input_size=(100, 100, 3), output_size=(3,),
                                           continuous=False)
        max_num_batches = 2
        config_dict = {
            'data_directories': self.info['episode_directories'],
            'output_path': self.output_dir,
            'random_seed': 1,
            'batch_size': 3
        }
        data_loader = DataLoader(config=DataLoaderConfig().create(config_dict=config_dict))
        data_loader.load_dataset()
        first_batch = []
        index = 0
        for index, batch in enumerate(data_loader.sample_shuffled_batch(max_number_of_batches=max_num_batches)):
            if index == 0:
                first_batch = deepcopy(batch)
            self.assertEqual(len(batch), config_dict['batch_size'])
        self.assertEqual(index, max_num_batches - 1)

        # test sampling seed for reproduction
        config_dict['random_seed'] = 2
        data_loader = DataLoader(config=DataLoaderConfig().create(config_dict=config_dict))
        data_loader.load_dataset()
        second_batch = []
        for index, batch in enumerate(data_loader.sample_shuffled_batch(max_number_of_batches=max_num_batches)):
            second_batch = deepcopy(batch)
            break
        self.assertNotEqual(np.sum(np.asarray(first_batch.observations[0])),
                            np.sum(np.asarray(second_batch.observations[0])))
        config_dict['random_seed'] = 1
        data_loader = DataLoader(config=DataLoaderConfig().create(config_dict=config_dict))
        data_loader.load_dataset()
        third_batch = []
        for index, batch in enumerate(data_loader.sample_shuffled_batch(max_number_of_batches=max_num_batches)):
            third_batch = deepcopy(batch)
            break
        self.assertEqual(np.sum(np.asarray(first_batch.observations[0])),
                         np.sum(np.asarray(third_batch.observations[0])))

    def test_calculate_weights_for_data_balancing_uniform(self):
        # uniform distributed case:
        data = np.random.uniform(0, 1, 1000)
        weights = calculate_weights(data)
        self.assertTrue(max(weights) - min(weights) < 0.3)

    def test_calculate_weights_for_data_balancing_normal(self):
        # normal distributed case:
        data = np.random.normal(0, 1, 1000)
        weights = calculate_weights(data)
        self.assertTrue(0.5 < max(weights) - min(weights) < 1.5)

    def test_calculate_weights_for_data_balancing_discrete(self):
        # discrete distributions:
        data = [0] * 30 + [1] * 70
        weights = calculate_weights(data)
        self.assertEqual(weights[0], weights[29])
        self.assertNotEqual(weights[29], weights[30])
        self.assertTrue(weights[0] - 2. < 0.1)
        self.assertTrue(weights[30] - 0.57157 < 0.1)

    def test_data_subsample(self):
        self.info = generate_dummy_dataset(self.data_saver, num_runs=20, input_size=(100, 100, 3), output_size=(3,),
                                           continuous=False)
        subsample = 4
        config_dict = {
            'data_directories': self.info['episode_directories'],
            'output_path': self.output_dir,
            'random_seed': 1,
            'batch_size': 3,
            'subsample': subsample
        }
        data_loader = DataLoader(config=DataLoaderConfig().create(config_dict=config_dict))
        data_loader.load_dataset()
        self.assertTrue(sum([np.ceil((el - 1) / subsample) + 1 for el in self.info['episode_lengths']]),
                        len(data_loader.get_dataset()))

    def test_select(self):
        data = list(range(10))
        indices = [3, 5]
        result = select(data, indices)
        self.assertEqual(result, [3, 5])

        data = torch.as_tensor([[v, 1, 10 - v] for v in range(10)])
        indices = [3, 5]
        result = select(data, indices)
        self.assertEqual((result - torch.as_tensor([[3, 1, 7], [5, 1, 5]])).sum().item(), 0)

        data = np.asarray([[v, 1, 10 - v] for v in range(10)])
        indices = [3, 5]
        result = select(data, indices)
        self.assertEqual((result - np.asarray([[3, 1, 7], [5, 1, 5]])).sum(), 0)

    def test_big_data_hdf5_loop(self):
        # create 3 datasets as hdf5 files
        hdf5_files = []
        infos = []
        for index in range(3):
            output_path = os.path.join(self.output_dir, f'ds{index}')
            os.makedirs(output_path, exist_ok=True)
            config_dict = {
                'output_path': output_path,
                'store_hdf5': True,
                'training_validation_split': 1.0
            }
            config = DataSaverConfig().create(config_dict=config_dict)
            self.data_saver = DataSaver(config=config)
            infos.append(generate_dummy_dataset(self.data_saver, num_runs=2, input_size=(3, 10, 10),
                                                fixed_input_value=(0.3 * index) * np.ones((3, 10, 10)), store_hdf5=True))
            self.assertTrue(os.path.isfile(os.path.join(output_path, 'train.hdf5')))
            hdf5_files.append(os.path.join(output_path, 'train.hdf5'))
            hdf5_files.append(os.path.join(output_path, 'wrong.hdf5'))

        # create data loader with big data tag and three hdf5 training sets
        conf = {'output_path': self.output_dir,
                'hdf5_files': hdf5_files,
                'batch_size': 15,
                'loop_over_hdf5_files': True}
        loader = DataLoader(DataLoaderConfig().create(config_dict=conf))

        # sample data batches and see that index increases every two batches sampled
        for batch in loader.get_data_batch():
            self.assertAlmostEqual(batch.observations[0][0, 0, 0].item(), 0)
        for batch in loader.get_data_batch():
            self.assertAlmostEqual(batch.observations[0][0, 0, 0].item(), 0.3, 2)
        for batch in loader.get_data_batch():
            self.assertAlmostEqual(batch.observations[0][0, 0, 0].item(), 0.6, 2)
        for batch in loader.get_data_batch():
            self.assertAlmostEqual(batch.observations[0][0, 0, 0].item(), 0, 2)
        for batch in loader.sample_shuffled_batch():
            self.assertAlmostEqual(batch.observations[0][0, 0, 0].item(), 0.3, 2)
        for batch in loader.sample_shuffled_batch():
            self.assertAlmostEqual(batch.observations[0][0, 0, 0].item(), 0.6, 2)
        for batch in loader.sample_shuffled_batch():
            self.assertAlmostEqual(batch.observations[0][0, 0, 0].item(), 0, 2)

    # def test_data_balancing(self): TODO
    #     # average action variance in batch with action balancing
    #     config = DataLoaderConfig().create(config_dict={
    #         'data_directories': self.info['episode_directories'],
    #         'output_path': self.output_dir,
    #         'balance_over_actions': True,
    #         'batch_size': 20
    #     })
    #     balanced_data_loader = DataLoader(config=config)
    #     balanced_data_loader.load_dataset(arrange_according_to_timestamp=False)
    #     action_variances_with_balancing = []
    #     for b_batch in balanced_data_loader.sample_shuffled_batch():
    #         action_variances_with_balancing.append(
    #             [np.std([a[dim] for a in b_batch.actions]) for dim in range(b_batch.actions[0].size()[0])]
    #         )
    #     action_variances_with_balancing = np.asarray(action_variances_with_balancing)
    #
    #     config = DataLoaderConfig().create(config_dict={
    #         'data_directories': self.info['episode_directories'],
    #         'output_path': self.output_dir,
    #         'balance_over_actions': False,
    #         'batch_size': 20
    #     })
    #     unbalanced_data_loader = DataLoader(config=config)
    #     unbalanced_data_loader.load_dataset(arrange_according_to_timestamp=False)
    #     action_variances_without_balancing = []
    #     for u_batch in unbalanced_data_loader.sample_shuffled_batch():
    #         action_variances_without_balancing.append(
    #             [np.std([a[dim] for a in u_batch.actions]) for dim in range(u_batch.actions[0].size()[0])]
    #         )
    #
    #     action_variances_without_balancing = np.asarray(action_variances_without_balancing)
    #
    #     self.assertGreaterEqual(np.mean(action_variances_with_balancing),
    #                             np.mean(action_variances_without_balancing))

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
