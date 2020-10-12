import shutil
import unittest
import os
from copy import deepcopy

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.data_loader import DataLoader, DataLoaderConfig
from src.core.utils import get_filename_without_extension
from src.data.data_saver import DataSaverConfig, DataSaver
from src.data.test.common_utils import generate_dummy_dataset
from src.data.utils import arrange_run_according_timestamps, calculate_weights, select, parse_binary_maps


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

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
