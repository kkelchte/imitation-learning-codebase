import shutil
import unittest
import os

import torch
import numpy as np

from src.data.data_loader import DataLoader, DataLoaderConfig
from src.data.data_saver import DataSaver, DataSaverConfig
from src.data.test.common_utils import experience_generator, generate_dummy_dataset
from src.core.data_types import TerminationType, Experience
from src.core.utils import get_filename_without_extension


class TestDataSaver(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        self.data_saver = None

    def test_experience_generator(self):
        for count, experience in enumerate(experience_generator()):
            if count == 0:
                self.assertEqual(experience.done, TerminationType.Unknown)
        self.assertTrue(experience.done in [TerminationType.Done, TerminationType.Success, TerminationType.Failure])

    def test_data_storage_in_raw_data(self):
        config_dict = {
            'output_path': self.output_dir,
        }
        config = DataSaverConfig().create(config_dict=config_dict)
        self.data_saver = DataSaver(config=config)
        info = generate_dummy_dataset(self.data_saver, num_runs=2)
        for total, episode_dir in zip(info['episode_lengths'], info['episode_directories']):
            self.assertEqual(len(os.listdir(os.path.join(self.output_dir, 'raw_data', episode_dir, 'observation'))),
                             total)
            with open(os.path.join(self.output_dir, 'raw_data', episode_dir, 'action.data')) as f:
                expert_controls = f.readlines()
                self.assertEqual(len(expert_controls), total)

    def test_data_storage_in_raw_data_with_data_size_limit(self):
        config_dict = {
            'output_path': self.output_dir,
            'max_size': 25
        }
        config = DataSaverConfig().create(config_dict=config_dict)
        self.data_saver = DataSaver(config=config)
        first_info = generate_dummy_dataset(self.data_saver, num_runs=2)
        self.assertEqual(sum(first_info['episode_lengths']), self.data_saver._frame_counter)
        self.data_saver.update_saving_directory()
        second_info = generate_dummy_dataset(self.data_saver, num_runs=2)
        self.assertTrue((sum(first_info['episode_lengths']) + sum(second_info['episode_lengths'])) >
                        config_dict['max_size'])
        self.assertTrue(self.data_saver._frame_counter <= config_dict['max_size'])
        raw_data_dir = os.path.dirname(self.data_saver.get_saving_directory())
        count_actual_frames = sum([len(os.listdir(os.path.join(raw_data_dir, episode_dir, 'observation')))
                                   for episode_dir in os.listdir(raw_data_dir)])
        self.assertEqual(count_actual_frames, self.data_saver._frame_counter)

    def test_create_train_validation_hdf5_files(self):
        num_runs = 10
        split = 0.7
        config_dict = {
            'output_path': self.output_dir,
            'training_validation_split': split,
            'store_hdf5': True
        }
        config = DataSaverConfig().create(config_dict=config_dict)
        self.data_saver = DataSaver(config=config)
        info = generate_dummy_dataset(self.data_saver, num_runs=num_runs)
        self.data_saver.create_train_validation_hdf5_files()

        config = DataLoaderConfig().create(config_dict={'output_path': self.output_dir,
                                                        'hdf5_files': [os.path.join(self.output_dir,
                                                                                    'train.hdf5')]})
        training_data_loader = DataLoader(config=config)
        training_data_loader.load_dataset()
        training_data = training_data_loader.get_dataset()

        config = DataLoaderConfig().create(config_dict={'output_path': self.output_dir,
                                                        'hdf5_files': [os.path.join(self.output_dir,
                                                                                    'validation.hdf5')]})
        validation_data_loader = DataLoader(config=config)
        validation_data_loader.load_dataset()
        validation_data = validation_data_loader.get_dataset()

        self.assertEqual(len(training_data), sum(info['episode_lengths'][:int(split * num_runs)]))
        self.assertEqual(len(validation_data), sum(info['episode_lengths'][int(split * num_runs):]))

    def test_create_hdf5_files_subsampled_in_time(self):
        num_runs = 10
        split = 1.0
        subsample = 3
        config_dict = {
            'output_path': self.output_dir,
            'training_validation_split': split,
            'store_hdf5': True,
            'subsample_hdf5': subsample
        }
        config = DataSaverConfig().create(config_dict=config_dict)
        self.data_saver = DataSaver(config=config)
        info = generate_dummy_dataset(self.data_saver, num_runs=num_runs)
        self.data_saver.create_train_validation_hdf5_files()

        config = DataLoaderConfig().create(config_dict={'output_path': self.output_dir,
                                                        'hdf5_files': [os.path.join(self.output_dir, 'train.hdf5')]})
        training_data_loader = DataLoader(config=config)
        training_data_loader.load_dataset()
        training_data = training_data_loader.get_dataset()

        self.assertEqual(len(training_data), sum([np.ceil((el - 1) / subsample) + 1 for el in info['episode_lengths']]))

    def test_empty_saving_directory(self):
        config_dict = {
            'output_path': self.output_dir
        }
        number_of_runs = 5
        config = DataSaverConfig().create(config_dict=config_dict)
        self.data_saver = DataSaver(config=config)
        info = generate_dummy_dataset(self.data_saver, num_runs=number_of_runs)
        self.assertEqual(len(os.listdir(os.path.join(self.output_dir, 'raw_data'))), number_of_runs)
        self.data_saver.empty_raw_data_in_output_directory()
        self.assertEqual(len(os.listdir(os.path.join(self.output_dir, 'raw_data'))), 0)

    def test_store_in_ram(self):
        config_dict = {
            'output_path': self.output_dir,
            'store_on_ram_only': True,
            'max_size': 10
        }
        number_of_runs = 10
        config = DataSaverConfig().create(config_dict=config_dict)
        self.data_saver = DataSaver(config=config)
        info = generate_dummy_dataset(self.data_saver, num_runs=number_of_runs)
        data = self.data_saver.get_dataset()
        self.assertEqual(len(data), config_dict['max_size'])
        for lst in [data.observations, data.actions, data.rewards, data.done]:
            self.assertEqual(len(lst), config_dict['max_size'])
            self.assertTrue(isinstance(lst[0], torch.Tensor))

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
