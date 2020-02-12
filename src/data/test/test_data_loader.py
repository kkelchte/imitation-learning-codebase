import shutil
import unittest
import os

import numpy as np
import torch

from src.data.data_types import Run
from src.data.dataset_loader import DataLoader, DataLoaderConfig, arrange_run_according_timestamps
from src.core.utils import get_filename_without_extension


def check_run_lengths(run: Run) -> bool:
    # assert all data streams have equal lengths, otherwise timestamp arranging failed
    stream_lengths = [
                         len(run.inputs[k]) for k in run.inputs.keys()
                     ] + [
                         len(run.outputs[k]) for k in run.outputs.keys()
                     ]
    return min(stream_lengths) == max(stream_lengths)


class TestDataLoader(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'test_dir/{get_filename_without_extension(__file__)}'
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        self.dummy_dataset = '/esat/opal/kkelchte/experimental_data/dummy_dataset'

    def test_arrange_run_according_timestamps(self):
        run = Run()
        time_stamps = {}
        # 9 values with only the third good to be kept
        run.inputs['sensor'] = torch.Tensor([[1], [1], [2]]*3)
        time_stamps['sensor'] = list(range(9))
        run.outputs['steering'] = torch.Tensor([[5]]*3)
        time_stamps['steering'] = [2+x*3 for x in range(3)]

        arranged_run = arrange_run_according_timestamps(run=run, time_stamps=time_stamps)
        self.assertTrue(check_run_lengths(run=arranged_run))
        # assert only correct values with corresponding control are kept
        self.assertTrue(sum(v == 2 for v in arranged_run.inputs['sensor']), len(arranged_run.inputs['sensor']))

    def test_data_storage_of_all_sensors(self):
        config_dict = {
            'data_directories': [os.path.join(self.dummy_dataset, 'raw_data', d)
                                 for d in os.listdir(os.path.join(self.dummy_dataset, 'raw_data'))],
            'output_path': self.output_dir,
            'inputs': ['forward_camera',
                       'current_waypoint'],
            'outputs': ['depth_scan',
                        'ros_expert']
        }
        config = DataLoaderConfig().create(config_dict=config_dict)
        data_loader = DataLoader(config=config)
        data_loader.load_dataset()

        # assert nothing is empty
        for x in config.inputs:
            self.assertTrue(x in data_loader.get_data()[0].inputs.keys())
            self.assertTrue(sum(data_loader.get_data()[0].inputs[x].shape) > 0)
        for y in config.outputs:
            self.assertTrue(y in data_loader.get_data()[0].outputs.keys())
            self.assertTrue(sum(data_loader.get_data()[0].outputs[y].shape) > 0)
        # assert lengths are equal
        for run in data_loader.get_data():
            self.assertTrue(check_run_lengths(run=run))

    def test_data_storage_with_input_sizes(self):
        config_dict = {
            'data_directories': [os.path.join(self.dummy_dataset, 'raw_data', d)
                                 for d in os.listdir(os.path.join(self.dummy_dataset, 'raw_data'))],
            'output_path': self.output_dir,
            'inputs': ['forward_camera',
                       'current_waypoint'],
            'outputs': ['depth_scan',
                        'ros_expert']
        }
        config = DataLoaderConfig().create(config_dict=config_dict)
        data_loader = DataLoader(config=config)
        data_loader.load_dataset(input_sizes=[[3, 64, 64], [1, 1, 2]])
        self.assertTrue(data_loader.get_data()[0].inputs['forward_camera'][0].size() == (3, 64, 64))
        self.assertTrue(data_loader.get_data()[0].inputs['current_waypoint'][0].size() == (1, 1, 2))

    def test_data_loader_with_relative_paths(self):
        config_dict = {'output_path': '/esat/opal/kkelchte/experimental_data/dummy_dataset',
                       'data_directories': ['raw_data/20-02-06_13-32-24',
                                            'raw_data/20-02-06_13-32-43'],
                       'inputs': ['forward_camera'],
                       'outputs': ['ros_expert']}
        config = DataLoaderConfig().create(config_dict=config_dict)
        for d in config.data_directories:
            self.assertTrue(os.path.isdir(d))
        data_loader = DataLoader(config=config)
        data_loader.load_dataset(input_sizes=[[3, 128, 128]],
                                 output_sizes=[[6]])
        self.assertTrue(len(data_loader.get_data()) != 0)

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
