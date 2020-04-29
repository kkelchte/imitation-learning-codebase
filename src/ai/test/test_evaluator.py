import os
import shutil
import unittest

import torch

from src.ai.base_net import InitializationType, ArchitectureConfig, BaseNet
from src.ai.evaluator import Evaluator, EvaluatorConfig
from src.core.utils import get_to_root_dir, get_filename_without_extension, generate_random_image
from src.ai.architectures import *  # Do not remove
from src.data.data_saver import DataSaverConfig, DataSaver
from src.data.test.common_utils import generate_dummy_dataset

evaluator_base_config = {
    "data_loader_config": {},
    "criterion": "MSELoss",
    "device": "cpu"
}

architecture_base_config = {
    "architecture": "tiny_128_rgb_1c",
    "load_checkpoint_dir": None,
    "initialisation_type": InitializationType.Xavier,
    "initialisation_seed": 0,
    "device": 'cpu',
}


class EvaluatorTest(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        evaluator_base_config['output_path'] = self.output_dir
        architecture_base_config['output_path'] = self.output_dir

    def test_evaluate_model_on_dataset(self):
        # generate network
        network = BaseNet(config=ArchitectureConfig().create(config_dict=architecture_base_config))

        # generate dummy dataset
        generate_random_dataset(output_dir=self.output_dir,
                                input_size=network.input_size,
                                output_size=network.output_size)
        config_dict = {
            'output_path': self.output_dir,
            'store_hdf5': True
        }
        config = DataSaverConfig().create(config_dict=config_dict)
        self.data_saver = DataSaver(config=config)
        self.info = generate_dummy_dataset(self.data_saver, num_runs=20)

        # generate evaluator
        evaluator = Evaluator(config=EvaluatorConfig().create(config_dict=evaluator_base_config),
                              network=network)

        # evaluate

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
