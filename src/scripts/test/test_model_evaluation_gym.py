import shutil
import unittest
import os

from src.scripts.experiment import ExperimentConfig, Experiment
from src.core.utils import get_filename_without_extension, get_to_root_dir

experiment_config = {
    "output_path": "/tmp",
    "data_saver_config": {},  # provide empty dict for default data_saving config, if None --> no data saved.
    "number_of_episodes": 2,
    "architecture_config": {
        "architecture": "tiny_128_rgb_6c",
        "initialisation_type": 'xavier',
        "random_seed": 0,
        "device": 'cpu'},
    "tensorboard": True,
    "environment_config": {
        "factory_key": "GYM",
        "max_number_of_steps": 5,
        "gym_config": {
            "random_seed": 123,
            "world_name": None,
            "render": False,
        },
    },
}


class TestGymModelEvaluation(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        shutil.rmtree(self.output_dir, ignore_errors=True)
        os.makedirs(self.output_dir, exist_ok=True)
        experiment_config['output_path'] = self.output_dir

    def test_discrete_stochastic_cart_pole(self):
        experiment_config["environment_config"]["gym_config"]["world_name"] = "CartPole-v0"
        experiment_config["architecture_config"]["architecture"] = "cart_pole_4_2d_stochastic"
        self.experiment = Experiment(ExperimentConfig().create(config_dict=experiment_config))
        self.experiment.run()
        raw_data_dirs = [os.path.join(self.output_dir, 'raw_data', d)
                         for d in os.listdir(os.path.join(self.output_dir, 'raw_data'))]
        self.assertEqual(len(raw_data_dirs), 1)
        run_dir = raw_data_dirs[0]
        with open(os.path.join(run_dir, 'done.data'), 'r') as f:
            self.assertEqual(experiment_config["number_of_episodes"] *
                             experiment_config["environment_config"]["max_number_of_steps"],
                             len(f.readlines()))
        self.experiment.shutdown()

    def test_continuous_stochastic_pendulum(self):
        experiment_config["environment_config"]["gym_config"]["world_name"] = "Pendulum-v0"
        experiment_config["architecture_config"]["architecture"] = "pendulum_3_1c_stochastic"
        self.experiment = Experiment(ExperimentConfig().create(config_dict=experiment_config))
        self.experiment.run()
        raw_data_dirs = [os.path.join(self.output_dir, 'raw_data', d)
                         for d in os.listdir(os.path.join(self.output_dir, 'raw_data'))]
        self.assertEqual(len(raw_data_dirs), 1)
        run_dir = raw_data_dirs[0]
        with open(os.path.join(run_dir, 'done.data'), 'r') as f:
            self.assertEqual(len(f.readlines()),
                             experiment_config["number_of_episodes"] *
                             experiment_config["environment_config"]["max_number_of_steps"])
        self.experiment.shutdown()

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
