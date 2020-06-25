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

    def test_save_and_restore_architecture(self):
        config_dict = {
            'output_path': self.output_dir,
            'number_of_epochs': 10,
            'number_of_episodes': -1,
            'train_every_n_steps': 10,
            'load_checkpoint_dir': None,
            'load_checkpoint_found': True,
            'tensorboard': False,
            'environment_config': {
                "factory_key": "GYM",
                "max_number_of_steps": 50,
                'normalize_observations': True,
                'normalize_rewards': True,
                'observation_clipping': 10,
                'reward_clipping': 10,
                "gym_config": {
                    "random_seed": 123,
                    "world_name": 'CartPole-v0',
                    "render": False,
                },
            },
            'data_saver_config': {
                'clear_buffer_before_episode': True,
                'store_on_ram_only': True
            },
            'architecture_config': {'architecture': 'cart_pole_4_2d_stochastic', 'log_std': 0.0, 'device': 'cpu',
                                    'initialisation_seed': 2048, 'initialisation_type': 'orthogonal'},
            'trainer_config': {
                'criterion': 'MSELoss',
                'critic_learning_rate': 0.0001,
                'actor_learning_rate': 0.00015,
                'scheduler_config': {'number_of_epochs': 488},
                'gradient_clip_norm': -1,
                'optimizer': 'Adam',
                'data_loader_config': {'batch_size': 64, 'data_sampling_seed': 2048},
                'device': 'cpu',
                'discount': 0.99,
                'factory_key': 'PPO',
                'gae_lambda': 0.95,
                'phi_key': 'gae',
                'entropy_coefficient': 0.0,
                'save_checkpoint_every_n': 50,
                'max_actor_training_iterations': 10,
                'max_critic_training_iterations': 10,
                'ppo_epsilon': 0.2,
                'kl_target': 0.01},
            'evaluator_config': None,
        }
        self.config = ExperimentConfig().create(
            config_dict=config_dict
        )
        experiment = Experiment(self.config)
        self.assertTrue(experiment._environment._observation_filter._mean is None)
        experiment.run()

        config_dict['architecture_config']['initialisation_seed'] = 543
        new_experiment = Experiment(ExperimentConfig().create(
            config_dict=config_dict
        ))
        # assert models are equal
        self.assertEqual(experiment._net.get_checksum(), new_experiment._net.get_checksum())
        # assert learning rate is loaded correctly
        self.assertEqual(experiment._trainer.actor_optimizer.lr, new_experiment._trainer.actor_optimizer.lr)
        # assert filter values are loaded correctly
        self.assertFalse(new_experiment._environment._observation_filter._mean is None)

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
