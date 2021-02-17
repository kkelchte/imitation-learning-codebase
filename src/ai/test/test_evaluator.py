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


class EvaluatorTest(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        evaluator_base_config['output_path'] = self.output_dir
        architecture_base_config['output_path'] = self.output_dir

    def test_evaluate_model_on_dataset(self):
        network = eval(architecture_base_config['architecture']).Net(
            config=ArchitectureConfig().create(config_dict=architecture_base_config)
        )
        info = generate_random_dataset_in_raw_data(output_dir=self.output_dir,
                                                   input_size=network.input_size,
                                                   output_size=network.output_size,
                                                   continuous=not network.discrete)

        # generate evaluator with correct data-loader
        evaluator_base_config['data_loader_config'] = {
            'data_directories': info['episode_directories'],
            'batch_size': 5
        }
        evaluator = Evaluator(config=EvaluatorConfig().create(config_dict=evaluator_base_config),
                              network=network)
        # evaluate
        error_msg = evaluator.evaluate()
        self.assertFalse('nan' in error_msg)

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
