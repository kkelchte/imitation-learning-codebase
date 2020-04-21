import shutil
import unittest
import os

from src.data.dataset_loader import DataLoaderConfig, DataLoader
from src.data.dataset_saver import DataSaver, DataSaverConfig
from src.data.test.common_utils import state_generator, generate_dummy_dataset
from src.sim.common.data_types import TerminalType
from src.core.utils import get_filename_without_extension


class TestDataSaver(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

    def test_data_storage_of_all_sensors(self):
        config_dict = {
            'output_path': self.output_dir
        }
        config = DataSaverConfig().create(config_dict=config_dict)
        data_saver = DataSaver(config=config)
        info = generate_dummy_dataset(data_saver, num_runs=1)
        total = info['episode_lengths'][0]
        episode_dir = info['episode_directories'][0]
        self.assertEqual(len(os.listdir(os.path.join(self.output_dir, 'raw_data', episode_dir, 'camera'))), total)
        with open(os.path.join(self.output_dir, 'raw_data', episode_dir, 'expert')) as f:
            expert_controls = f.readlines()
            self.assertEqual(len(expert_controls), total)

    def test_data_storage_of_one_sensor_in_custom_place(self):
        config_dict = {
            'output_path': self.output_dir,
            'saving_directory': os.path.join(self.output_dir, 'custom_place'),
            'sensors': ['camera']
        }
        config = DataSaverConfig().create(config_dict=config_dict)
        data_saver = DataSaver(config=config)
        info = generate_dummy_dataset(data_saver, num_runs=1)
        total = info['episode_lengths'][0]
        episode_dir = info['episode_directories'][0]
        self.assertEqual(episode_dir, os.path.join(self.output_dir, 'custom_place'))
        self.assertEqual(len(os.listdir(os.path.join(self.output_dir, 'custom_place', 'camera'))), total)
        with open(os.path.join(self.output_dir, 'custom_place', 'expert')) as f:
            expert_controls = f.readlines()
            self.assertEqual(len(expert_controls), total)
        self.assertTrue(not os.path.exists(os.path.join(self.output_dir, 'custom_place', 'depth')))

    # def test_clear_data_saving_directory(self):

    def test_create_train_validation_hdf5_files(self):
        config_dict = {
            'output_path': self.output_dir,
            'training_validation_split': 1.0,
            'store_hdf5': True
        }
        config = DataSaverConfig().create(config_dict=config_dict)
        data_saver = DataSaver(config=config)
        info = generate_dummy_dataset(data_saver)
        episode_lengths = info['episode_lengths']
        episode_directories = info['episode_directories']
        data_saver.create_train_validation_hdf5_files()

        config_dict = {'output_path': self.output_dir,
                       'hdf5_file': 'train.hdf5',
                       'inputs': ['camera'],
                       'outputs': ['expert']}
        config = DataLoaderConfig().create(config_dict=config_dict)
        data_loader = DataLoader(config=config)
        data_loader.load_dataset()
        data = data_loader.get_data()
        self.assertEqual(len(data), len(episode_lengths))
        for index, run in enumerate(data):
            self.assertEqual(len(run), episode_lengths[index])
            count = len(os.listdir(os.path.join(episode_directories[index], 'camera')))
            self.assertEqual(len(run), count)

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
            'store_on_ram_only': True
        }
        number_of_runs = 5
        config = DataSaverConfig().create(config_dict=config_dict)
        data_saver = DataSaver(config=config)
        info = generate_dummy_dataset(data_saver, num_runs=number_of_runs)
        print(info)
        data_saver.data_set
        self.assertEqual(len(os.listdir(os.path.join(self.output_dir, 'raw_data'))), number_of_runs)
        data_saver.empty_raw_data_in_output_directory()
        self.assertEqual(len(os.listdir(os.path.join(self.output_dir, 'raw_data'))), 0)

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
