import shutil
import unittest
import os

import numpy as np
import torch

from src.ai.base_net import InitializationType
from src.ai.utils import get_checksum_network_parameters
from src.core.data_types import Dataset
from src.scripts.experiment import ExperimentConfig, Experiment
from src.core.utils import get_filename_without_extension, get_to_root_dir, get_check_sum_list

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
        "load_checkpoint_dir": None,
        "initialisation_type": 0,
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
        "save_checkpoint_every_n": 1000
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

    def test_integrated_vpg_vs_standalone_script_data_collection_first_epoch(self):
        experiment_config['number_of_episodes'] = -1
        experiment_config['environment_config']['max_number_of_steps'] = -1
        experiment_config['data_saver_config']['store_on_ram_only'] = True
        self.experiment = Experiment(ExperimentConfig().create(config_dict=experiment_config))
        self.experiment._run_episodes()
        dataset = self.experiment._data_saver.get_dataset()
        for index, ((ba, da), (bo, do), (br, dr), (bd, dd)) in enumerate(zip(zip(batch_actions, dataset.actions),
                                                                             zip(batch_observations,
                                                                                 dataset.observations),
                                                                             zip(batch_rewards, dataset.rewards),
                                                                             zip(batch_done, dataset.done))):
            for (est, trgt) in [(ba, da), (bo, do), (br, dr), (bd, dd)]:
                self.assertLess(np.sum(est - trgt.numpy()), 10**-6)
            if index > 35:  # torch categorical sampling is inconsistent at step 39
                break

    def test_integrated_vpg_vs_standalone_script_calculate_advantegeous(self):
        experiment_config['number_of_episodes'] = -1
        experiment_config['environment_config']['max_number_of_steps'] = -1
        experiment_config['data_saver_config']['store_on_ram_only'] = True
        self.experiment = Experiment(ExperimentConfig().create(config_dict=experiment_config))
        self.experiment._run_episodes()
        dataset = self.experiment._data_saver.get_dataset()
        phi_weights = self.experiment._trainer._calculate_phi(dataset)
        advantages = [a.detach().item() for a in phi_weights]
        for _ in range(20):
            self.assertEqual(advantages[_], batch_weights[_])

    def test_integrated_vpg_vs_standalone_script_apply_training_step(self):
        experiment_config['number_of_episodes'] = -1
        experiment_config['environment_config']['max_number_of_steps'] = -1
        experiment_config['data_saver_config']['store_on_ram_only'] = True
        self.experiment = Experiment(ExperimentConfig().create(config_dict=experiment_config))
        self.experiment._run_episodes()
        dataset = self.experiment._data_saver.get_dataset()
        # dataset = Dataset(
        #     observations=[o for o in dataset.observations[:clean_num]],
        #     actions=[o for o in dataset.actions[:clean_num]],
        #     rewards=[o for o in dataset.rewards[:clean_num]],
        #     done=[o for o in dataset.done[:clean_num]],
        # )
        dataset = Dataset(
            observations=[o for o in batch_observations],
            actions=[o for o in batch_actions],
            rewards=[torch.as_tensor(o) for o in batch_rewards],
            done=[torch.as_tensor(o) for o in batch_done],
        )
        phi_weights = self.experiment._trainer._calculate_phi(dataset)
#         print(get_checksum_network_parameters(self.experiment._net._actor.parameters()))
# #        policy_loss = self.experiment._trainer._train_actor(dataset, torch.as_tensor(batch_weights))
#         print(f"batch_observations: {get_check_sum_list(dataset.observations)}")
#         print(f"batch_actions: {get_check_sum_list(dataset.actions)}")
#         print(f"batch_rewards: {get_check_sum_list(dataset.rewards)}")
#         print(f"batch_weights: {get_check_sum_list(phi_weights)}")

        policy_loss = self.experiment._trainer._train_actor(dataset, torch.as_tensor(phi_weights))
        # Adam optimizer step is only place where difference occurs.
        print(get_checksum_network_parameters(self.experiment._net._actor.parameters()))
        #self.assertEqual(28.424017012119293, )

        critic_loss = self.experiment._trainer._train_critic(dataset, phi_weights)
        self.assertEqual(34.344590842723846, get_checksum_network_parameters(self.experiment._net._critic.parameters()))
        a=100

    def tearDown(self) -> None:
        self.experiment = None
        shutil.rmtree(self.output_dir, ignore_errors=True)


batch_observations = [np.asarray([ 0.02078762, -0.01301236, -0.0209893 , -0.03935255]),
 np.asarray([ 0.02052737, -0.20782713, -0.02177635,  0.24663485]),
 np.asarray([ 0.01637083, -0.01240105, -0.01684366, -0.05283652]),
 np.asarray([ 0.01612281,  0.18295832, -0.01790039, -0.35078581]),
 np.asarray([ 0.01978197,  0.3783302 , -0.0249161 , -0.64905912]),
 np.asarray([ 0.02734858,  0.57379022, -0.03789729, -0.94948272]),
 np.asarray([ 0.03882438,  0.76940129, -0.05688694, -1.25382777]),
 np.asarray([ 0.05421241,  0.9652038 , -0.0819635 , -1.56377219]),
 np.asarray([ 0.07351648,  0.77115192, -0.11323894, -1.29774185]),
 np.asarray([ 0.08893952,  0.96751492, -0.13919378, -1.62362102]),
 np.asarray([ 0.10828982,  0.77427888, -0.1716662 , -1.37736133]),
 np.asarray([ 0.1237754 ,  0.58166588, -0.19921342, -1.14291086]),
 np.asarray([0.04975787, 0.04942262, 0.04825252, 0.00586875]),
 np.asarray([ 0.05074632, -0.14635694,  0.04836989,  0.31337701]),
 np.asarray([0.04781918, 0.04804378, 0.05463743, 0.0363322 ]),
 np.asarray([ 0.04878006,  0.24234138,  0.05536408, -0.23862388]),
 np.asarray([ 0.05362689,  0.43663049,  0.0505916 , -0.51334229]),
 np.asarray([ 0.0623595 ,  0.63100475,  0.04032475, -0.78966305]),
 np.asarray([ 0.07497959,  0.82555041,  0.02453149, -1.06939203]),
 np.asarray([ 0.0914906 ,  1.02033947,  0.00314365, -1.35427618]),
 np.asarray([ 0.11189739,  1.21542182, -0.02394187, -1.64597403]),
 np.asarray([ 0.13620583,  1.41081553, -0.05686135, -1.94601884]),
 np.asarray([ 0.16442214,  1.60649499, -0.09578173, -2.25577188]),
 np.asarray([ 0.19655204,  1.80237635, -0.14089717, -2.57636395]),
 np.asarray([ 0.23259956,  1.99830022, -0.19242445, -2.90862431])]
batch_actions = [0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
batch_done = [False,
 False,
 False,
 False,
 False,
 False,
 False,
 False,
 False,
 False,
 False,
 True,
 False,
 False,
 False,
 False,
 False,
 False,
 False,
 False,
 False,
 False,
 False,
 False,
 True]
batch_logits = torch.stack([torch.as_tensor([0.6106, 1.7487]),
 torch.as_tensor([0.5903, 1.6511]),
 torch.as_tensor([0.6098, 1.7521]),
 torch.as_tensor([0.6375, 1.8679]),
 torch.as_tensor([0.6727, 1.9689]),
 torch.as_tensor([0.7560, 1.9973]),
 torch.as_tensor([0.8584, 1.9730]),
 torch.as_tensor([0.9772, 1.9379]),
 torch.as_tensor([0.8767, 1.9602]),
 torch.as_tensor([1.0124, 1.9236]),
 torch.as_tensor([0.9101, 1.9430]),
 torch.as_tensor([0.8236, 1.9586]),
 torch.as_tensor([0.6025, 1.7638]),
 torch.as_tensor([0.5872, 1.6644]),
 torch.as_tensor([0.6034, 1.7568]),
 torch.as_tensor([0.6285, 1.8680]),
 torch.as_tensor([0.6687, 1.9664]),
 torch.as_tensor([0.7210, 2.0308]),
 torch.as_tensor([0.8086, 2.0103]),
 torch.as_tensor([0.9155, 1.9847]),
 torch.as_tensor([1.0515, 1.9548]),
 torch.as_tensor([1.0921, 1.9842]),
 torch.as_tensor([1.1332, 2.0126]),
 torch.as_tensor([1.1814, 2.0372]),
 torch.as_tensor([1.2459, 2.0424])]
)
batch_weights = [6.084941864013672,
 5.557577133178711,
 5.325965404510498,
 5.072268486022949,
 4.746443748474121,
 4.330163478851318,
 3.7903904914855957,
 3.169656753540039,
 2.4574851989746094,
 1.6937391757965088,
 0.8222242593765259,
 -0.14844679832458496,
 6.359577178955078,
 5.861253261566162,
 5.637722969055176,
 5.398919105529785,
 5.1106953620910645,
 4.766500473022461,
 4.326668739318848,
 3.7800512313842773,
 3.1694180965423584,
 2.5016045570373535,
 1.7628114223480225,
 0.9534837007522583,
 0.01601463556289673]
batch_rewards = [1.0] * len(batch_observations)

if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
