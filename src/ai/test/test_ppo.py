import os
import shutil
import unittest

from src.ai.base_net import InitializationType, ArchitectureConfig
from src.ai.ppo import ProximatePolicyGradient
from src.ai.trainer import TrainerConfig
from src.ai.utils import generate_random_dataset_in_raw_data, get_checksum_network_parameters
from src.core.utils import get_to_root_dir, get_filename_without_extension
from src.ai.architectures import *  # Do not remove


trainer_base_config = {
    "data_loader_config": {},
    "criterion": "MSELoss",
    "device": "cpu"
}

architecture_base_config = {
    "architecture": "cart_pole_4_2d_stochastic",
    "load_checkpoint_dir": None,
    "initialisation_type": InitializationType.Xavier,
    "initialisation_seed": 0,
    "device": 'cpu',
}


class PPOTest(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        architecture_base_config['output_path'] = self.output_dir
        trainer_base_config['output_path'] = self.output_dir
        self.network = eval(architecture_base_config['architecture']).Net(
            config=ArchitectureConfig().create(config_dict=architecture_base_config)
        )
        # checksum network

        info = generate_random_dataset_in_raw_data(output_dir=self.output_dir,
                                                   num_runs=5,
                                                   input_size=self.network.input_size,
                                                   output_size=self.network.output_size,
                                                   continuous=not self.network.discrete)
        trainer_base_config['data_loader_config'] = {
            'data_directories': info['episode_directories'],
        }

    def test_ppo_gae(self):
        before_check_actor = get_checksum_network_parameters(self.network.get_actor_parameters())
        before_check_critic = get_checksum_network_parameters(self.network.get_critic_parameters())
        trainer_base_config['phi_key'] = 'gae'
        trainer = ProximatePolicyGradient(config=TrainerConfig().create(config_dict=trainer_base_config),
                                          network=self.network)
        trainer.train()
        # test if network has changed
        self.assertNotEqual(get_checksum_network_parameters(self.network.get_actor_parameters()), before_check_actor)
        self.assertNotEqual(get_checksum_network_parameters(self.network.get_critic_parameters()), before_check_critic)

    def test_ppo_return(self):
        before_check_actor = get_checksum_network_parameters(self.network.get_actor_parameters())
        before_check_critic = get_checksum_network_parameters(self.network.get_critic_parameters())
        trainer_base_config['phi_key'] = 'return'
        trainer = ProximatePolicyGradient(config=TrainerConfig().create(config_dict=trainer_base_config),
                                          network=self.network)
        trainer.train()
        # test if network has changed
        self.assertNotEqual(get_checksum_network_parameters(self.network.get_actor_parameters()), before_check_actor)
        self.assertNotEqual(get_checksum_network_parameters(self.network.get_critic_parameters()), before_check_critic)

    def test_ppo_reward_to_go(self):
        before_check_actor = get_checksum_network_parameters(self.network.get_actor_parameters())
        before_check_critic = get_checksum_network_parameters(self.network.get_critic_parameters())
        trainer_base_config['phi_key'] = 'reward-to-go'
        trainer = ProximatePolicyGradient(config=TrainerConfig().create(config_dict=trainer_base_config),
                                          network=self.network)
        trainer.train()
        # test if network has changed
        self.assertNotEqual(get_checksum_network_parameters(self.network.get_actor_parameters()), before_check_actor)
        self.assertNotEqual(get_checksum_network_parameters(self.network.get_critic_parameters()), before_check_critic)

    def test_ppo_value_baseline(self):
        before_check_actor = get_checksum_network_parameters(self.network.get_actor_parameters())
        before_check_critic = get_checksum_network_parameters(self.network.get_critic_parameters())
        trainer_base_config['phi_key'] = 'value-baseline'
        trainer = ProximatePolicyGradient(config=TrainerConfig().create(config_dict=trainer_base_config),
                                          network=self.network)
        trainer.train()
        # test if network has changed
        self.assertNotEqual(get_checksum_network_parameters(self.network.get_actor_parameters()), before_check_actor)
        self.assertNotEqual(get_checksum_network_parameters(self.network.get_critic_parameters()), before_check_critic)

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
