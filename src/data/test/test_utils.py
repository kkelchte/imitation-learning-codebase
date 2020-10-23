import shutil
import unittest
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.data_loader import DataLoader, DataLoaderConfig
from src.core.utils import get_filename_without_extension
from src.data.data_saver import DataSaverConfig, DataSaver
from src.data.test.common_utils import generate_dummy_dataset
from src.data.utils import arrange_run_according_timestamps, calculate_weights, select, parse_binary_maps, \
    create_random_gradient_image


class TestUtil(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        config_dict = {
            'output_path': self.output_dir,
            'store_hdf5': True
        }
        config = DataSaverConfig().create(config_dict=config_dict)
        self.data_saver = DataSaver(config=config)
        self.info = generate_dummy_dataset(self.data_saver, num_runs=20, input_size=(100, 100, 3), output_size=(3,),
                                           continuous=False)

    @unittest.skip
    def test_parse_binary_map(self):
        line_image = torch.ones((3, 100, 100))
        line_image[0:2, 40:43, :] = 0
        result = parse_binary_maps([line_image])
        plt.imshow(result[0])
        plt.show()

        line_image = torch.ones((3, 100, 100))
        line_image[0:2, 40:43, :] = 0
        result = parse_binary_maps([line_image], invert=True)
        plt.imshow(result[0])
        plt.show()
        self.assertTrue(True)

    def test_joint_generators(self):
        def generate_a(num: int = 5):
            for _ in range(num):
                yield _

        def generate_b(num: int = 3):
            for _ in range(num):
                yield _

        for a, b in zip(generate_a(), generate_b()):
            self.assertEqual(a, b)

    @unittest.skip
    def test_create_random_gradient_image(self):
        n = 5
        fig, axes = plt.subplots(1, n + 1)
        for axe_index in range(n):
            ob = axes[axe_index].imshow(
                create_random_gradient_image(size=(100, 100, 3), dtype="np.float32",
                                             low=0., high=1.), vmin=0, vmax=1)
            axes[axe_index].axis('off')
        fig.colorbar(ob, cax=axes[n], orientation='horizontal')
        plt.show()
        self.assertTrue(True)

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
