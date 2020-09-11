import os
import shutil
import time
import unittest
from glob import glob

import numpy as np

from src.ai.utils import generate_random_dataset_in_raw_data
from src.core.utils import get_to_root_dir, get_filename_without_extension, get_data_dir
from src.data.data_loader import DataLoader, DataLoaderConfig
from src.scripts.data_cleaning import DataCleaner, DataCleaningConfig


class DatacleaningTest(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{get_data_dir(os.environ["HOME"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

    def test_create_dataset_and_clean(self):
        info = generate_random_dataset_in_raw_data(output_dir=self.output_dir,
                                                   num_runs=20,
                                                   input_size=(100, 100, 3),
                                                   output_size=(1,),
                                                   continuous=True,
                                                   store_hdf5=False)
        cleaner_config_dict = {
            'output_path': self.output_dir,
            'data_loader_config': {
                'data_directories': info['episode_directories'],
                'input_size': (150, 150, 1)
            },
            'training_validation_split': 0.7,
        }
        data_cleaner = DataCleaner(config=DataCleaningConfig().create(config_dict=cleaner_config_dict))
        data_cleaner.clean()
        data_loader_train = DataLoader(config=DataLoaderConfig().create(config_dict={
            'output_path': self.output_dir,
            'hdf5_files': glob(f'{self.output_dir}/train*.hdf5')
        }))
        data_loader_train.load_dataset()
        data_loader_validation = DataLoader(config=DataLoaderConfig().create(config_dict={
            'output_path': self.output_dir,
            'hdf5_files': glob(f'{self.output_dir}/validation*.hdf5')
        }))
        data_loader_validation.load_dataset()
        ratio = len(data_loader_train.get_dataset())/(0. + len(data_loader_train.get_dataset()) +
                                                      len(data_loader_validation.get_dataset()))
        self.assertTrue(ratio > 0.6)
        self.assertTrue(ratio < 0.8)

    def test_split_hdf5_chunks(self):
        info = generate_random_dataset_in_raw_data(output_dir=self.output_dir,
                                                   num_runs=20,
                                                   input_size=(100, 100, 3),
                                                   output_size=(1,),
                                                   continuous=True,
                                                   store_hdf5=False)
        cleaner_config_dict = {
            'output_path': self.output_dir,
            'data_loader_config': {
                'data_directories': info['episode_directories'],
            },
            'training_validation_split': 1.0,
            'max_hdf5_size': 5*10**6
        }
        data_cleaner = DataCleaner(config=DataCleaningConfig().create(config_dict=cleaner_config_dict))
        data_cleaner.clean()
        for hdf5_file in glob(f'{self.output_dir}/train*.hdf5'):
            data_loader = DataLoader(config=DataLoaderConfig().create(config_dict={
                'output_path': self.output_dir,
                'hdf5_files': [hdf5_file]
            }))
            data_loader.load_dataset()
            self.assertTrue(data_loader.get_dataset().get_memory_size() < 6*10**6)

    def test_clip_first_x_frames(self):
        info = generate_random_dataset_in_raw_data(output_dir=self.output_dir,
                                                   num_runs=20,
                                                   input_size=(100, 100, 3),
                                                   output_size=(1,),
                                                   continuous=True,
                                                   store_hdf5=False)
        cleaner_config_dict = {
            'output_path': self.output_dir,
            'data_loader_config': {
                'data_directories': info['episode_directories'],
            },
            'training_validation_split': 1.0,
            'remove_first_n_timestamps': 5
        }
        data_cleaner = DataCleaner(config=DataCleaningConfig().create(config_dict=cleaner_config_dict))
        data_cleaner.clean()
        data_loader = DataLoader(config=DataLoaderConfig().create(config_dict={
            'output_path': self.output_dir,
            'hdf5_files': glob(f'{self.output_dir}/train*.hdf5')
        }))
        data_loader.load_dataset()
        self.assertEqual(sum(e for e in info['episode_lengths']) - len(info['episode_lengths']) * 5,
                         len(data_loader.get_dataset()))

    @unittest.skip
    def test_line_world_preprocessing(self):
        line_image = np.ones((100, 100, 3))
        line_image[40:43, :, 0:2] = 0
        info = generate_random_dataset_in_raw_data(output_dir=self.output_dir,
                                                   num_runs=20,
                                                   input_size=(100, 100, 3),
                                                   output_size=(1,),
                                                   continuous=True,
                                                   fixed_input_value=line_image,
                                                   store_hdf5=False)
        cleaner_config_dict = {
            'output_path': self.output_dir,
            'data_loader_config': {
                'data_directories': info['episode_directories'],
            },
            'training_validation_split': 0.7,
            'remove_first_n_timestamps': 5,
            'binary_maps_as_target': True,
            'augment_background_noise': True
        }
        data_cleaner = DataCleaner(config=DataCleaningConfig().create(config_dict=cleaner_config_dict))
        data_cleaner.clean()
        data_loader = DataLoader(config=DataLoaderConfig().create(config_dict={
            'output_path': self.output_dir,
            'hdf5_files': glob(f'{self.output_dir}/train*.hdf5')
        }))
        data_loader.load_dataset()
        import matplotlib.pyplot as plt
        plt.imshow(data_loader.get_dataset().observations[0].permute(1, 2, 0))
        plt.show()
        plt.imshow(data_loader.get_dataset().actions[0])
        plt.show()

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)
        time.sleep(2)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
