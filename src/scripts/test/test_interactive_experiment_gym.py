import shutil
import time
import unittest
import os

import yaml

from src.scripts.interactive_experiment import InteractiveExperimentConfig, InteractiveExperiment
from src.core.utils import get_filename_without_extension

base_config = {
    "output_path": "/tmp",
    "number_of_episodes": 2,
    "environment_config": {
      "factory_key": 1,  # gym
      "max_number_of_steps": 100,
      "gym_config": {
          'random_seed': 123,
          'world_name': 'CartPole-v0',
          'render': True,
        },
      },
    "data_saver_config": {
        "store_on_ram_only": True
    },
    "actor_config": {
        'name':  '',
        'model_config': {
            'architecture': {

            }
        }
    }
}


class TestGymInteractive(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        base_config['output_path'] = self.output_dir

    def test_data_collection(self):
        self.config = InteractiveExperimentConfig().create(
            config_dict=base_config
        )
        self.experiment = InteractiveExperiment(self.config)
        self.experiment.run()
        time.sleep(1)

    # def test_model_evaluation(self):
    #     self.config = InteractiveExperimentConfig().create(
    #         config_dict=base_config
    #     )
    #     self.experiment = InteractiveExperiment(self.config)
    #     self.experiment.run()

    def tearDown(self) -> None:
        self.experiment.shutdown()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
