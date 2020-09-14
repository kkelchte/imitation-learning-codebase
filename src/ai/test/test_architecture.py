import os
import shutil
import unittest
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np

from src.ai.base_net import ArchitectureConfig, BaseNet
from src.ai.utils import mlp_creator, generate_random_dataset_in_raw_data
from src.core.utils import get_to_root_dir, get_filename_without_extension, generate_random_image
from src.ai.architectures import *  # Do not remove

base_config = {
    "architecture": "",
    "initialisation_type": 'xavier',
    "random_seed": 0,
    "device": 'cpu',
}


class ArchitectureTest(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        base_config['output_path'] = self.output_dir

    def test_tiny_128_rgb_6c_initialisation_store_load(self):
        # Test initialisation of different seeds
        base_config['architecture'] = 'tiny_128_rgb_6c'
        base_config['initialisation_type'] = 'xavier'
        network = eval(base_config['architecture']).Net(
            config=ArchitectureConfig().create(config_dict=base_config),
        )
        for p in network.parameters():
            check_network = p.data
            break
        base_config['random_seed'] = 2
        second_network = eval(base_config['architecture']).Net(
            config=ArchitectureConfig().create(config_dict=base_config)
        )
        for p in second_network.parameters():
            check_second_network = p.data
            break
        self.assertNotEqual(torch.sum(check_second_network), torch.sum(check_network))
        base_config['random_seed'] = 0
        third_network = eval(base_config['architecture']).Net(
            config=ArchitectureConfig().create(config_dict=base_config)
        )
        for p in third_network.parameters():
            check_third_network = p.data
            break
        self.assertEqual(torch.sum(check_third_network), torch.sum(check_network))

        # test storing and reloading
        checkpoint = second_network.get_checkpoint()
        network.load_checkpoint(checkpoint)
        for p in network.parameters():
            check_network = p.data
            break
        self.assertEqual(torch.sum(check_second_network), torch.sum(check_network))
        self.assertNotEqual(torch.sum(check_third_network), torch.sum(check_network))

    def test_tiny_128_rgb_6c_preprocessed_and_raw_input(self):
        base_config['architecture'] = 'tiny_128_rgb_6c'
        network = eval(base_config['architecture']).Net(
            config=ArchitectureConfig().create(config_dict=base_config)
        )
        # test normal batch of data
        self.assertEqual(network.forward(torch.randint(255, (10, *network.input_size))/255.).shape[1:],
                         network.output_size)

        # test single unprocessed data point
        processed_inputs = super(type(network), network).forward(torch.randint(255, (network.input_size[1],
                                                                                     network.input_size[2],
                                                                                     network.input_size[0])),
                                                                 train=False)
        self.assertTrue(processed_inputs.max() <= 1)
        self.assertTrue(processed_inputs.min() >= 0)
        self.assertEqual(len(processed_inputs.shape), len(network.input_size) + 1)
        self.assertEqual(processed_inputs.shape[2], network.input_size[1])
        self.assertEqual(processed_inputs.shape[3], network.input_size[2])

    def test_tiny_128_rgb_6c_raw_input_zero_centered(self):
        base_config['architecture'] = 'tiny_128_rgb_6c'
        network = eval(base_config['architecture']).Net(
            config=ArchitectureConfig().create(config_dict=base_config)
        )
        network.input_scope = 'zero_centered'
        # test single unprocessed data point
        processed_inputs = super(type(network), network).forward(torch.randint(255, (network.input_size[1],
                                                                                     network.input_size[2],
                                                                                     network.input_size[0])),
                                                                 train=False)
        self.assertTrue(processed_inputs.max() <= 1)
        self.assertTrue(-1 <= processed_inputs.min() < 0)
        self.assertEqual(len(processed_inputs.shape), len(network.input_size) + 1)
        self.assertEqual(processed_inputs.shape[2], network.input_size[1])
        self.assertEqual(processed_inputs.shape[3], network.input_size[2])

    def test_mlp_creator(self):
        network = mlp_creator(sizes=[4, 10, 10, 1],
                              activation=nn.ReLU(),
                              output_activation=None,
                              bias_in_last_layer=False)
        self.assertEqual(len(network), 5)
        count = 0
        for p in network.parameters():
            count += np.prod(p.shape)
        self.assertEqual(count, 170)

    def test_initialisation(self):
        base_config['architecture'] = 'cart_pole_4_2d_stochastic'
        base_config['initialisation_type'] = 'constant'
        network = eval(base_config['architecture']).Net(
            config=ArchitectureConfig().create(config_dict=base_config)
        )
        for p in network._actor[0].parameters():
            if len(p.shape) == 1:
                self.assertEqual(torch.min(p), 0)
                self.assertEqual(torch.max(p), 0)
            else:
                self.assertEqual(torch.min(p), 0.03)
                self.assertEqual(torch.max(p), 0.03)

    def test_sidetuned_network(self):
        base_config['architecture'] = 'dronet_sidetuned'
        network = eval(base_config['architecture']).Net(
            config=ArchitectureConfig().create(config_dict=base_config)
        )
        fixed_weight_checksum = network.conv2d_1.weight.data.sum().item()
        variable_weight_checksum = network.sidetune_conv2d_1.weight.data.sum().item()
        alpha_value = network.alpha.item()
        optimizer = torch.optim.Adam(network.parameters())
        for i in range(10):
            optimizer.zero_grad()
            network.forward(torch.randn(network.input_size)).mean().backward()
            optimizer.step()
        self.assertEqual(fixed_weight_checksum, network.conv2d_1.weight.data.sum().item())
        self.assertNotEqual(variable_weight_checksum, network.sidetune_conv2d_1.weight.data.sum().item())
        self.assertNotEqual(alpha_value, network.alpha.item())

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
