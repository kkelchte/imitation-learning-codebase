import os
import shutil
import unittest

import yaml

from src.core.utils import get_filename_without_extension
from src.scripts.dataset_experiment import DatasetExperimentConfig, DatasetExperiment


class TestDatasetExperiment(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        with open(f'src/scripts/test/config/test_dataset_experiment_config.yml', 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        config_dict['output_path'] = self.output_dir
        with open(os.path.join(self.output_dir, 'config.yml'), 'w') as f:
            yaml.dump(config_dict, f)
        self.config = DatasetExperimentConfig().create(
            config_file=os.path.join(self.output_dir, 'config.yml')
        )

    def test_run(self):
        experiment = DatasetExperiment(config=self.config)
        experiment.run()

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
