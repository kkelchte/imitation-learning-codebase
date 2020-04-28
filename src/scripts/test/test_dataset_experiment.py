import os
import shutil
import unittest

from src.core.utils import get_filename_without_extension
from src.scripts.dataset_experiment import DatasetExperimentConfig, DatasetExperiment

base_config = {
  "output_path": "/esat/opal/kkelchte/experimental_data/dummy_dataset",
  "number_of_epochs": 1,
  "model_config": {
    "device": "cpu",
    "architecture": "tiny_net_v0"
  },
  "evaluator_config": {
    "device": "cpu",
    "criterion": "MSELoss",
    "data_loader_config": {
      "inputs": [
        "forward_camera"
      ],
      "data_directories": [
        "/esat/opal/kkelchte/experimental_data/dummy_dataset/raw_data/20-02-06_13-32-43"
      ],
      "outputs": [
        "ros_expert"
      ]
    }
  },
  "trainer_config": {
    "optimizer": "SGD",
    "data_loader_config": {
      "balance_targets": True,
      "inputs": [
        "forward_camera"
      ],
      "data_directories": [
        "/esat/opal/kkelchte/experimental_data/dummy_dataset/raw_data/20-02-06_13-32-24"
      ],
      "outputs": [
        "ros_expert"
      ]
    },
    "device": "cuda",
    "criterion": "MSELoss",
    "learning_rate": 0.01,
    "batch_size": 32
  }
}


class TestDatasetExperiment(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        base_config['output_path'] = self.output_dir
        self.config = DatasetExperimentConfig().create(
            config_dict=base_config
        )

    def test_run(self):
        experiment = DatasetExperiment(config=self.config)
        experiment.run()
        self.assertTrue(False)

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
