import os
import shutil
import unittest

import numpy as np

from src.ai.base_net import ArchitectureConfig
from src.ai.evaluator import Evaluator, EvaluatorConfig
from src.ai.utils import generate_random_dataset_in_raw_data
from src.core.utils import get_to_root_dir, get_filename_without_extension
from src.ai.architectures import *  # Do not remove
from src.data.data_loader import DataLoaderConfig, DataLoader

evaluator_base_config = {
    "data_loader_config": {},
    "criterion": "MSELoss",
    "device": "cpu"
}

architecture_base_config = {
    "architecture": "tiny_128_rgb_6c",
    "initialisation_type": 'xavier',
    "random_seed": 0,
    "device": 'cpu',
}


class DummyDatasetGeneratorTest(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        evaluator_base_config['output_path'] = self.output_dir
        architecture_base_config['output_path'] = self.output_dir

    def test_generate_random_dataset_in_raw_data(self):
        num_runs = 10
        # generate network
        network = eval(architecture_base_config['architecture']).Net(
            config=ArchitectureConfig().create(config_dict=architecture_base_config)
        )

        # generate dummy dataset
        info = generate_random_dataset_in_raw_data(output_dir=self.output_dir,
                                                   num_runs=num_runs,
                                                   input_size=network.input_size,
                                                   output_size=network.output_size,
                                                   continuous=not network.discrete,)
        data_loader_config = {
            'output_path': self.output_dir,
            'data_directories': info['episode_directories'],
        }
        data_loader = DataLoader(config=DataLoaderConfig().create(config_dict=data_loader_config))
        data_loader.load_dataset()
        self.assertEqual(sum(d != 0 for d in data_loader.get_dataset().done),
                         num_runs)

    def test_generate_random_dataset_with_train_validation_hdf5(self):
        num_runs = 10
        # generate network
        network = eval(architecture_base_config['architecture']).Net(
            config=ArchitectureConfig().create(config_dict=architecture_base_config)
        )

        # generate dummy dataset
        info = generate_random_dataset_in_raw_data(output_dir=self.output_dir,
                                                   num_runs=num_runs,
                                                   input_size=network.input_size,
                                                   output_size=network.output_size,
                                                   continuous=not network.discrete,
                                                   store_hdf5=True)
        data_loader_config = {
            'output_path': self.output_dir,
            'hdf5_file': 'train.hdf5'
        }
        data_loader = DataLoader(config=DataLoaderConfig().create(config_dict=data_loader_config))
        data_loader.load_dataset()
        self.assertNotEqual(sum(d != 0 for d in data_loader.get_dataset().done), 0)

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
