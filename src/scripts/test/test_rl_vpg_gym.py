import shutil
import unittest
import os

from src.ai.base_net import InitializationType
from src.scripts.experiment import ExperimentConfig, Experiment
from src.core.utils import get_filename_without_extension, get_to_root_dir

experiment_config = {
    "output_path": "/tmp",
    "number_of_epochs": 5,
    "number_of_episodes": 2,
    "environment_config": {
        "factory_key": "GYM",
        "max_number_of_steps": 10,
        "gym_config": {
            "random_seed": 123,
            "world_name": "CartPole-v0",
            "render": False,
        },
    },
    "data_saver_config": {},  # provide empty dict for default data_saving config, if None --> no data saved.
    "architecture_config": {
        "architecture": "cart_pole_4_2d_stochastic",
        "load_checkpoint_dir": None,
        "initialisation_type": InitializationType.Xavier,
        "initialisation_seed": 0,
        "device": 'cpu',
    },
    "trainer_config": {
        "factory_key": "VPG",
        "data_loader_config": {},
        "criterion": "MSELoss",
        "device": "cpu",
        "phi_key": "gae",
        "gae_lambda": 0.5,
        "discount": 0.99
    },
    "tensorboard": False,
}


class TestVPGGym(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        shutil.rmtree(self.output_dir, ignore_errors=True)
        os.makedirs(self.output_dir, exist_ok=True)
        experiment_config['output_path'] = self.output_dir

    def test_vpg_cart_pole_fs(self):
        self.experiment = Experiment(ExperimentConfig().create(config_dict=experiment_config))
        self.experiment.run()
        self.experiment.shutdown()

    def test_vpg_cart_pole_ram(self):
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
