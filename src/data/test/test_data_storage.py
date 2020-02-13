import shutil
import unittest
import os

from src.data.dataset_saver import DataSaver, DataSaverConfig
from src.data.test.common_utils import state_generator
from src.sim.common.data_types import TerminalType
from src.core.utils import get_filename_without_extension


class TestDataStorage(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'test_dir/{get_filename_without_extension(__file__)}'
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

    def test_data_storage_of_all_sensors(self):
        config_dict = {
            'output_path': self.output_dir
        }
        config = DataSaverConfig().create(config_dict=config_dict)
        data_saver = DataSaver(config=config)
        total = 0
        for state in state_generator():
            if state.terminal != TerminalType.Unknown:
                total += 1
            data_saver.save(state=state,
                            action=None)
        episode_dir = os.listdir(os.path.join(self.output_dir, 'raw_data'))[0]
        self.assertEqual(len(os.listdir(os.path.join(self.output_dir, 'raw_data', episode_dir, 'rgb'))), total)
        with open(os.path.join(self.output_dir, 'raw_data', episode_dir, 'expert')) as f:
            expert_controls = f.readlines()
            self.assertEqual(len(expert_controls), total)

    def test_data_storage_of_one_sensor(self):
        config_dict = {
            'output_path': self.output_dir,
            'saving_directory': os.path.join(self.output_dir, 'custom_place'),
            'sensors': ['rgb']
        }
        config = DataSaverConfig().create(config_dict=config_dict)
        data_saver = DataSaver(config=config)
        total = 0
        for state in state_generator():
            if state.terminal != TerminalType.Unknown:
                total += 1
            data_saver.save(state=state,
                            action=None)
        self.assertEqual(len(os.listdir(os.path.join(self.output_dir, 'custom_place', 'rgb'))), total)
        with open(os.path.join(self.output_dir, 'custom_place', 'expert')) as f:
            expert_controls = f.readlines()
            self.assertEqual(len(expert_controls), total)
        self.assertTrue(not os.path.exists(os.path.join(self.output_dir, 'custom_place', 'depth')))

    def test_create_train_validation_hdf5_files(self):
        config_dict = {
            'output_path': self.output_dir,
            'sensors': ['rgb']
        }
        config = DataSaverConfig().create(config_dict=config_dict)
        data_saver = DataSaver(config=config)
        total = 0
        for state in state_generator():
            if state.terminal != TerminalType.Unknown:
                total += 1
            data_saver.save(state=state,
                            action=None)
        data_saver.create_train_validation_hdf5_files()

        print('finished')

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
