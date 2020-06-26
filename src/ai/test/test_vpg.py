import os
import shutil
import unittest

from src.ai.base_net import ArchitectureConfig
from src.ai.trainer import TrainerConfig
from src.ai.utils import generate_random_dataset_in_raw_data, get_checksum_network_parameters
from src.ai.vpg import VanillaPolicyGradient
from src.core.utils import get_to_root_dir, get_filename_without_extension
from src.ai.architectures import *  # Do not remove
from src.data.data_loader import DataLoaderConfig, DataLoader


trainer_base_config = {
    "data_loader_config": {},
    "criterion": "MSELoss",
    "device": "cpu"
}

architecture_base_config = {
    "architecture": "cart_pole_4_2d_stochastic",
    "initialisation_type": 'xavier',
    "random_seed": 0,
    "device": 'cpu',
}


class VPGTest(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        architecture_base_config['output_path'] = self.output_dir
        trainer_base_config['output_path'] = self.output_dir

    def test_actor_critic_gae(self):
        network = eval(architecture_base_config['architecture']).Net(
            config=ArchitectureConfig().create(config_dict=architecture_base_config)
        )
        # checksum network
        before_check = get_checksum_network_parameters(network.parameters())
        info = generate_random_dataset_in_raw_data(output_dir=self.output_dir,
                                                   num_runs=5,
                                                   input_size=network.input_size,
                                                   output_size=network.output_size,
                                                   continuous=not network.discrete)
        trainer_base_config['data_loader_config'] = {
            'data_directories': info['episode_directories'],
        }
        trainer_base_config['phi_key'] = 'gae'
        trainer = VanillaPolicyGradient(config=TrainerConfig().create(config_dict=trainer_base_config),
                                        network=network)
        trainer.train()
        # test if network has changed
        after_check = get_checksum_network_parameters(network.parameters())
        self.assertNotEqual(before_check, after_check)

    def test_actor_critic_value_baseline(self):
        network = eval(architecture_base_config['architecture']).Net(
            config=ArchitectureConfig().create(config_dict=architecture_base_config)
        )
        # checksum network
        before_check = get_checksum_network_parameters(network.parameters())
        info = generate_random_dataset_in_raw_data(output_dir=self.output_dir,
                                                   num_runs=5,
                                                   input_size=network.input_size,
                                                   output_size=network.output_size,
                                                   continuous=not network.discrete)
        trainer_base_config['data_loader_config'] = {
            'data_directories': info['episode_directories'],
        }
        trainer_base_config['phi_key'] = 'value-baseline'
        trainer = VanillaPolicyGradient(config=TrainerConfig().create(config_dict=trainer_base_config),
                                        network=network)
        trainer.train()
        # test if network has changed
        after_check = get_checksum_network_parameters(network.parameters())
        self.assertNotEqual(before_check, after_check)

    def test_actor_critic_separate_training(self):
        network = eval(architecture_base_config['architecture']).Net(
            config=ArchitectureConfig().create(config_dict=architecture_base_config)
        )
        # checksum network
        before_check = get_checksum_network_parameters(network.parameters())
        info = generate_random_dataset_in_raw_data(output_dir=self.output_dir,
                                                   num_runs=5,
                                                   input_size=network.input_size,
                                                   output_size=network.output_size,
                                                   continuous=not network.discrete)
        before_check_actor = get_checksum_network_parameters(network.get_actor_parameters())
        before_check_critic = get_checksum_network_parameters(network.get_critic_parameters())

        trainer_base_config['data_loader_config'] = {
            'data_directories': info['episode_directories'],
        }
        trainer_base_config['phi_key'] = 'gae'
        trainer = VanillaPolicyGradient(config=TrainerConfig().create(config_dict=trainer_base_config),
                                        network=network)
        batch = trainer.data_loader.get_dataset()
        phi_weights = trainer._calculate_phi(batch, values=trainer._net.critic(batch.observations))
        trainer._train_actor(batch, phi_weights)
        self.assertNotEqual(get_checksum_network_parameters(network.get_actor_parameters()), before_check_actor)
        self.assertEqual(get_checksum_network_parameters(network.get_critic_parameters()), before_check_critic)
        trainer._train_critic(batch, phi_weights)
        self.assertNotEqual(get_checksum_network_parameters(network.get_critic_parameters()), before_check_critic)

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
