import shutil
import unittest
import os

import numpy as np

#from src.data.dataset_loader import DataLoaderConfig, DataLoader
import torch

from src.data.dataset_saver import DataSaver, DataSaverConfig
from src.data.test.common_utils import experience_generator, generate_dummy_dataset  # , generate_dummy_dataset
from src.sim.common.data_types import TerminationType
from src.core.utils import get_filename_without_extension


class TestDataSaver(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        #saving_directory_tag: str = ''
        #saving_directory: str = None
        #sensors: List[str] = None
        #actors: List[str] = None
        #training_validation_split: float = 0.9
        #store_hdf5: bool = False
        #store_on_ram_only: bool = True

    def test_experience_generator(self):
        observation = np.random.randint((100, 100, 3), dtype=np.uint8)
        action = np.random.uniform(-1, 1)
        reward = np.random.uniform(-1, 1)
        for count, experience in enumerate(experience_generator(observation=observation,
                                                                action=action,
                                                                reward=reward)):
            if count == 0:
                self.assertEqual(experience.done, TerminationType.Unknown)
            self.assertEqual(np.sum(experience.observation), np.sum(observation))
            self.assertEqual(experience.action, action)
            self.assertEqual(experience.reward, reward)
        self.assertTrue(experience.done in [TerminationType.Done, TerminationType.Success, TerminationType.Failure])

    def test_data_storage_in_raw_data(self):
        config_dict = {
            'output_path': self.output_dir,
        }
        config = DataSaverConfig().create(config_dict=config_dict)
        data_saver = DataSaver(config=config)
        info = generate_dummy_dataset(data_saver, num_runs=2)
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
        data_saver = DataSaver(config=config)
        first_info = generate_dummy_dataset(data_saver, num_runs=2)
        self.assertEqual(sum(first_info['episode_lengths']), data_saver._frame_counter)
        second_info = generate_dummy_dataset(data_saver, num_runs=2)
        self.assertTrue((sum(first_info['episode_lengths']) + sum(second_info['episode_lengths'])) >
                        config_dict['max_size'])
        self.assertTrue(data_saver._frame_counter <= config_dict['max_size'])
        raw_data_dir = os.path.dirname(data_saver.get_saving_directory())
        count_actual_frames = sum([len(os.listdir(os.path.join(raw_data_dir, episode_dir, 'observation')))
                                   for episode_dir in os.listdir(raw_data_dir)])
        self.assertEqual(count_actual_frames, data_saver._frame_counter)

    def test_create_train_validation_hdf5_files(self):
        config_dict = {
            'output_path': self.output_dir,
            'training_validation_split': 0.5,
            'store_hdf5': True
        }
        config = DataSaverConfig().create(config_dict=config_dict)
        data_saver = DataSaver(config=config)
        info = generate_dummy_dataset(data_saver, num_runs=10)
        episode_lengths = info['episode_lengths']
        episode_directories = info['episode_directories']
        data_saver.create_train_validation_hdf5_files()

        # config_dict = {'output_path': self.output_dir,
        #                'hdf5_file': 'train.hdf5'}
        # config = DataLoaderConfig().create(config_dict=config_dict)
        # training_data_loader = DataLoader(config=config)
        # training_data_loader.load_dataset()
        # training_data = training_data_loader.get_data()
        #
        # config_dict = {'output_path': self.output_dir,
        #                'hdf5_file': 'validation.hdf5'}
        # config = DataLoaderConfig().create(config_dict=config_dict)
        # validation_data_loader = DataLoader(config=config)
        # validation_data_loader.load_dataset()
        # validation = validation_data_loader.get_data()

        self.assertTrue(False)
        # self.assertEqual(len(data), len(episode_lengths))
        # for index, run in enumerate(data):
        #     self.assertEqual(len(run), episode_lengths[index])
        #     count = len(os.listdir(os.path.join(episode_directories[index], 'camera')))
        #     self.assertEqual(len(run), count)

    def test_empty_saving_directory(self):
        config_dict = {
            'output_path': self.output_dir
        }
        number_of_runs = 5
        config = DataSaverConfig().create(config_dict=config_dict)
        data_saver = DataSaver(config=config)
        info = generate_dummy_dataset(data_saver, num_runs=number_of_runs)
        print(info)
        self.assertEqual(len(os.listdir(os.path.join(self.output_dir, 'raw_data'))), number_of_runs)
        data_saver.empty_raw_data_in_output_directory()
        self.assertEqual(len(os.listdir(os.path.join(self.output_dir, 'raw_data'))), 0)

    def test_store_in_ram(self):
        config_dict = {
            'output_path': self.output_dir,
            'store_on_ram_only': True,
            'max_size': 10
        }
        number_of_runs = 5
        config = DataSaverConfig().create(config_dict=config_dict)
        data_saver = DataSaver(config=config)
        info = generate_dummy_dataset(data_saver, num_runs=number_of_runs)
        print(info)
        data = data_saver.get_dataset()
        self.assertEqual(len(data), config_dict['max_size'])
        for lst in [data.observations, data.actions, data.rewards, data.done]:
            self.assertEqual(len(lst), config_dict['max_size'])
            self.assertTrue(isinstance(lst[0], torch.Tensor))

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
