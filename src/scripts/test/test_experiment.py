import os
import shutil
import unittest

from src.core.utils import get_filename_without_extension
from src.scripts.experiment import ExperimentConfig, Experiment


class TestExperiment(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

    def test_experiment_with_empty_config(self):
        self.config = ExperimentConfig().create(
            config_dict={
                'output_path': self.output_dir
            }
        )
        self.experiment = Experiment(self.config)
        self.experiment.run()
        self.experiment.shutdown()

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
