import os
import shutil
import unittest
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np

from src.ai.base_net import ArchitectureConfig, BaseNet
from src.ai.utils import mlp_creator, generate_random_dataset_in_raw_data
from src.core.utils import get_to_root_dir, get_filename_without_extension, generate_random_image, get_data_dir
from src.ai.architectures import *  # Do not remove

base_config = {
    "architecture": "",
    "initialisation_type": 'xavier',
    "random_seed": 0,
    "device": 'cpu',
}

# checksum of weights of network layers from KERAS model
conv2d = -5.570542335510254
batch_normalization = 19.45431900024414
conv2d_1 = -25.008115768432617
batch_normalization_1 = 14.739563941955566
conv2d_3 = -5.242682456970215
conv2d_2 = -27.535648345947266
batch_normalization_2 = 20.360254287719727
conv2d_4 = -11.319987297058105
batch_normalization_3 = 29.241863250732422
conv2d_6 = 12.120674133300781
conv2d_5 = -29.784086227416992
batch_normalization_4 = 27.909969329833984
conv2d_7 = -38.84811019897461
batch_normalization_5 = 57.77732849121094
conv2d_9 = 53.0998420715332
conv2d_8 = -48.354522705078125
dense_1 = -78.36251831054688
dense = 4.674846649169922


class LoadCheckpointTest(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        base_config['output_path'] = self.output_dir

    def test_dronenet_random_input(self):
        base_config['architecture'] = 'dronet'
        network = eval(base_config['architecture']).Net(
            config=ArchitectureConfig().create(config_dict=base_config),
        )
        print(network)
        print(network(torch.randn(network.input_size)))

    def test_load_dronet_checkpoint(self):
        base_config['architecture'] = 'dronet'
        network = eval(base_config['architecture']).Net(
            config=ArchitectureConfig().create(config_dict=base_config),
        )
        checkpoint = torch.load(os.path.join(os.environ['PWD'], 'experimental_data', 'pretrained_models', 'dronet',
                                             'torch_checkpoints', 'checkpoint_latest.ckpt'))
        network.load_checkpoint(checkpoint['net_ckpt'])
        self.assertLess(network.conv2d_1.weight.sum().item() - conv2d, 0.001)
        self.assertLess(network.batch_normalization_1.weight.sum().item() - batch_normalization, 0.001)
        self.assertLess(network.conv2d_2.weight.sum().item() - conv2d_1, 0.001)
        self.assertLess(network.batch_normalization_2.weight.sum().item() - batch_normalization_1, 0.001)
        self.assertLess(network.conv2d_4.weight.sum().item() - conv2d_3, 0.001)
        self.assertLess(network.conv2d_3.weight.sum().item() - conv2d_2, 0.001)
        self.assertLess(network.batch_normalization_3.weight.sum().item() - batch_normalization_2, 0.001)
        self.assertLess(network.conv2d_5.weight.sum().item() - conv2d_4, 0.001)
        self.assertLess(network.batch_normalization_4.weight.sum().item() - batch_normalization_3, 0.001)
        self.assertLess(network.conv2d_7.weight.sum().item() - conv2d_6, 0.001)
        self.assertLess(network.conv2d_6.weight.sum().item() - conv2d_5, 0.001)
        self.assertLess(network.batch_normalization_5.weight.sum().item() - batch_normalization_4, 0.001)
        self.assertLess(network.conv2d_8.weight.sum().item() - conv2d_7, 0.001)
        self.assertLess(network.batch_normalization_6.weight.sum().item() - batch_normalization_5, 0.001)
        self.assertLess(network.conv2d_10.weight.sum().item() - conv2d_9, 0.001)
        self.assertLess(network.conv2d_9.weight.sum().item() - conv2d_8, 0.001)

    def test_load_imagenet_pretrained_checkpoint(self):
        base_config['architecture'] = 'auto_encoder_deeply_supervised'
        network = eval(base_config['architecture']).Net(
            config=ArchitectureConfig().create(config_dict=base_config),
        )
        checkpoint = torch.load(os.path.join(get_data_dir(os.environ['PWD']), 'experimental_data', 'pretrained_models',
                                             'auto_encoder_deeply_supervised', 'torch_checkpoints',
                                             'checkpoint_latest.ckpt'))
        network.load_checkpoint(checkpoint['net_ckpt'])

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()

