import os
import shutil
import unittest
from copy import deepcopy, copy

import torch
import torch.nn as nn
import numpy as np

from src.ai.base_net import ArchitectureConfig, BaseNet
from src.ai.trainer import TrainerConfig
from src.ai.trainer_factory import TrainerFactory
from src.ai.utils import mlp_creator, generate_random_dataset_in_raw_data
from src.core.utils import get_to_root_dir, get_filename_without_extension, generate_random_image
from src.ai.architectures import *  # Do not remove
from src.data.test.common_utils import generate_dataset

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

        network.remove()
        second_network.remove()
        third_network.remove()

    def test_tiny_128_rgb_6c_preprocessed_and_raw_input(self):
        base_config['architecture'] = 'tiny_128_rgb_6c'
        network = eval(base_config['architecture']).Net(
            config=ArchitectureConfig().create(config_dict=base_config)
        )
        # test normal batch of data
        self.assertEqual(network.forward(torch.randint(255, (10, *network.input_size))/255.).shape[1:],
                         network.output_size)

        # test single unprocessed data point
        processed_inputs = network.process_inputs(torch.randint(255, (network.input_size[1],
                                                                      network.input_size[2],
                                                                      network.input_size[0])),
                                                  train=False)
        self.assertTrue(processed_inputs.max() <= 1)
        self.assertTrue(processed_inputs.min() >= 0)
        self.assertEqual(len(processed_inputs.shape), len(network.input_size) + 1)
        self.assertEqual(processed_inputs.shape[2], network.input_size[1])
        self.assertEqual(processed_inputs.shape[3], network.input_size[2])
        network.remove()

    def test_tiny_128_rgb_6c_raw_input_zero_centered(self):
        base_config['architecture'] = 'tiny_128_rgb_6c'
        network = eval(base_config['architecture']).Net(
            config=ArchitectureConfig().create(config_dict=base_config)
        )
        network.input_scope = 'zero_centered'
        # test single unprocessed data point
        processed_inputs = network.process_inputs(torch.randint(255, (network.input_size[1],
                                                                      network.input_size[2],
                                                                      network.input_size[0])),
                                                  train=False)
        self.assertTrue(processed_inputs.max() <= 1)
        self.assertTrue(-1 <= processed_inputs.min() < 0)
        self.assertEqual(len(processed_inputs.shape), len(network.input_size) + 1)
        self.assertEqual(processed_inputs.shape[2], network.input_size[1])
        self.assertEqual(processed_inputs.shape[3], network.input_size[2])
        network.remove()

    def test_input_output_architectures(self):
        for vae in [False, True]:
            for architecture in ['auto_encoder_conv1', 'auto_encoder_conv3', 'auto_encoder_conv3_deep',
                                 'auto_encoder_unet', 'auto_encoder_conv1_200', 'auto_encoder_conv3_200',
                                 'auto_encoder_conv3_deep_200', 'auto_encoder_conv5_200',
                                 'auto_encoder_conv5_deep_200']:
                base_config['architecture'] = architecture
                base_config['vae'] = vae
                network = eval(base_config['architecture']).Net(
                    config=ArchitectureConfig().create(config_dict=base_config)
                )
                # test single unprocessed data point
                batch_size = 15
                outputs = network.forward(torch.randint(255, (batch_size,
                                                              network.input_size[0],
                                                              network.input_size[1],
                                                              network.input_size[2])), train=False)
                self.assertEqual(outputs.shape[0], batch_size)
                for i, v in enumerate(network.output_size):
                    print(f'observed output size {outputs.shape[1+i]} is equal to expected {v}')
                    self.assertEqual(outputs.shape[1+i], v)
                network.remove()

    @unittest.skip
    def test_training_through_vae(self):
        # Failing test
        base_config['architecture'] = 'auto_encoder_unet'
        base_config['vae'] = False
        base_config['initialisation_type'] = 'constant'
        network = eval(base_config['architecture']).Net(
            config=ArchitectureConfig().create(config_dict=base_config)
        )
        # test single unprocessed data point
        batch_size = 1
        initial_weight = sum(p.data.sum() for k, p in network.named_parameters() if 'in_conv' in k).item()

        optimizer = torch.optim.SGD(network.parameters(), 0.1)
        for i in range(3):
            optimizer.zero_grad()
            outputs = network.forward(i * torch.randn((batch_size,
                                                      network.input_size[0],
                                                      network.input_size[1],
                                                      network.input_size[2])), train=True).abs().sum()
            print(outputs)
            outputs.backward()
            optimizer.step()

        final_weight = sum(p.data.sum() for k, p in network.named_parameters() if 'in_conv' in k).item()
        print(f'initial {initial_weight} -> final: {final_weight}')
        self.assertNotEqual(initial_weight, final_weight)
        network.remove()

    def test_gradients_through_index(self):
        a = torch.randn((10, 1024), requires_grad=True)
        b = torch.nn.Linear(1024, 512)
        c = torch.nn.Linear(256, 1)
        d = torch.nn.Linear(256, 1)
        o = b(a)
        o1 = o[:, :256]
        o2 = o[:, 256:]
        out = (c(o1) + d(o2)).mean()
        out.backward()
        self.assertNotEqual(sum(p.grad.sum() for p in b.parameters()).item(), 0)
        self.assertNotEqual(sum(p.grad.sum() for p in c.parameters()).item(), 0)
        self.assertNotEqual(sum(p.grad.sum() for p in d.parameters()).item(), 0)

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
        network.remove()

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
        network.remove()

    def test_auto_encoder_deeply_supervised(self):
        for arch in ['auto_encoder_deeply_supervised_share_weights', 'auto_encoder_deeply_supervised_2layered',
                     'auto_encoder_deeply_supervised', 'auto_encoder_deeply_supervised_maxpool']:
            base_config['architecture'] = arch
            base_config['initialisation_type'] = 'xavier'
            network = eval(base_config['architecture']).Net(
                config=ArchitectureConfig().create(config_dict=base_config)
            )

            # test single unprocessed data point
            network.forward(torch.randn((10, *network.input_size)), train=True)

            initial_parameters = deepcopy(dict(network.named_parameters()))

            # test trainer
            trainer_config = {
                'output_path': self.output_dir,
                'optimizer': 'Adam',
                'learning_rate': 0.01,
                'factory_key': 'DeepSupervision',
                'data_loader_config': {},
                'criterion': 'WeightedBinaryCrossEntropyLoss',
                "criterion_args_str": 'beta=0.9',
            }
            torch.autograd.set_detect_anomaly(True)
            trainer = TrainerFactory().create(config=TrainerConfig().create(config_dict=trainer_config), network=network)
            dataset = generate_dataset(input_size=network.input_size,
                                       output_size=network.output_size)
            trainer.data_loader.set_dataset(dataset)
            trainer.train()
            for k, p in network.named_parameters():
                print(f'{k}: {initial_parameters[k].sum().item()} <-> {p.sum().item()}')
                self.assertNotEqual(initial_parameters[k].sum().item(), p.sum().item())
            network.remove()

    def test_auto_encoder_deeply_supervised_with_confidence(self):
        for arch in ['auto_encoder_deeply_supervised_confidence']:
            base_config['architecture'] = arch
            base_config['initialisation_type'] = 'xavier'
            network = eval(base_config['architecture']).Net(
                config=ArchitectureConfig().create(config_dict=base_config)
            )

            # test single unprocessed data point
            network.forward(torch.randn((10, *network.input_size)), train=True)

            initial_parameters = deepcopy(dict(network.named_parameters()))

            # test trainer
            trainer_config = {
                'output_path': self.output_dir,
                'optimizer': 'Adam',
                'learning_rate': 0.01,
                'factory_key': 'DeepSupervisionConfidence',
                'data_loader_config': {},
                'criterion': 'WeightedBinaryCrossEntropyLoss',
                "criterion_args_str": 'beta=0.9',
            }
            trainer = TrainerFactory().create(config=TrainerConfig().create(config_dict=trainer_config), network=network)
            dataset = generate_dataset(input_size=network.input_size,
                                       output_size=network.output_size)
            trainer.data_loader.set_dataset(dataset)
            trainer.train()
            for k, p in network.named_parameters():
                print(f'{k}: {initial_parameters[k].sum().item()} <-> {p.sum().item()}')
                self.assertNotEqual(initial_parameters[k].sum().item(), p.sum().item())
            network.remove()

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
