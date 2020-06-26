import shutil
import unittest
import os

from src.scripts.experiment import ExperimentConfig, Experiment
from src.core.utils import get_filename_without_extension, get_to_root_dir

experiment_config = {
    "output_path": "/tmp",
    "number_of_epochs": 5,
    "number_of_episodes": -1,
    "environment_config": {
        "factory_key": "GYM",
        "max_number_of_steps": -1,
        "gym_config": {
            "random_seed": 123,
            "world_name": "CartPole-v0",
            "render": False,
        },
    },
    "data_saver_config": {
        "store_on_ram_only": True,
        "clear_buffer_before_episode": True,
    },
    "architecture_config": {
        "architecture": "cart_pole_4_2d_stochastic",
        "initialisation_type": 'xavier',
        "initialisation_seed": 123,
        "device": 'cpu',
    },
    "trainer_config": {
        "factory_key": "VPG",
        "data_loader_config": {
            "batch_size": 200
        },
        "criterion": "MSELoss",
        "optimizer": "Adam",
        "device": "cpu",
        "phi_key": "gae",
        "gae_lambda": 0.95,
        "discount": 0.95,
    },
    "tensorboard": False,
}


class TestVPGGym(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        experiment_config['output_path'] = self.output_dir

    def test_vpg_cart_pole_fs(self):
        experiment_config['number_of_episodes'] = 2
        experiment_config['data_saver_config']['store_on_ram_only'] = False
        experiment_config['data_saver_config']['separate_raw_data_runs'] = True
        self.experiment = Experiment(ExperimentConfig().create(config_dict=experiment_config))
        self.experiment.run()
        self.experiment.shutdown()

    def test_vpg_cart_pole_ram(self):
        experiment_config['number_of_episodes'] = 2
        experiment_config['data_saver_config']['store_on_ram_only'] = True
        self.experiment = Experiment(ExperimentConfig().create(config_dict=experiment_config))
        self.experiment.run()
        self.experiment.shutdown()

    def tearDown(self) -> None:
        self.experiment = None
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
