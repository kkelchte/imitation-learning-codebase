import shutil
import unittest
import os

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.data.data_types import Run
from src.data.dataset_loader import DataLoader, DataLoaderConfig, arrange_run_according_timestamps
from src.core.utils import get_filename_without_extension
from src.data.dataset_saver import DataSaverConfig, DataSaver
from src.data.test.common_utils import generate_dummy_dataset
from src.data.utils import calculate_probabilities, get_ideal_number_of_bins, calculate_probabilites_per_run


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
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        config_dict = {
            'output_path': self.output_dir,
            'store_hdf5': True
        }
        config = DataSaverConfig().create(config_dict=config_dict)
        self.data_saver = DataSaver(config=config)
        self.info = generate_dummy_dataset(self.data_saver, num_runs=1)

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
            'data_directories': self.info['episode_directories'],
            'output_path': self.output_dir,
            'inputs': self.info['inputs'],
            'outputs': self.info['outputs']
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
            'data_directories': self.info['episode_directories'],
            'output_path': self.output_dir,
            'inputs': self.info['inputs'],
            'outputs': self.info['outputs']
        }
        config = DataLoaderConfig().create(config_dict=config_dict)
        data_loader = DataLoader(config=config)
        input_sizes = [[3, 64, 64], [1, 1, 360]]
        data_loader.load_dataset(input_sizes=input_sizes)
        for input_type, input_size in zip(self.info['inputs'], input_sizes):
            self.assertEqual(data_loader.get_data()[0].inputs[input_type][0].size(), torch.Size(input_size))

    def test_data_loader_with_relative_paths(self):
        config_dict = {
            'data_directories': ['raw_data/' + os.path.basename(p) for p in self.info['episode_directories']],
            'output_path': self.output_dir,
            'inputs': self.info['inputs'],
            'outputs': self.info['outputs']
        }
        config = DataLoaderConfig().create(config_dict=config_dict)
        data_loader = DataLoader(config=config)
        data_loader.load_dataset()

        config = DataLoaderConfig().create(config_dict=config_dict)
        for d in config.data_directories:
            self.assertTrue(os.path.isdir(d))

    @unittest.skip("calculating probabilities has inapropriate dependency on ros_expert")
    def test_data_loaders_data_balancing(self):
        config_dict = {
            'data_directories': self.info['episode_directories'],
            'output_path': self.output_dir,
            'inputs': self.info['inputs'],
            'outputs': self.info['outputs']
        }
        config = DataLoaderConfig().create(config_dict=config_dict)
        data_loader = DataLoader(config=config)
        data_loader.load_dataset()

        # test calculate probabilities for run
        data = [float(d) for d in data_loader._dataset.data[0].outputs['expert'][:, 5]]
        probabilities = calculate_probabilities(data)

        # by sampling new data with probabilities and asserting difference among histogram bins is relatively low
        clean_data = []
        for i in range(300):
            clean_data.append(np.random.choice(data, p=probabilities))
        y, x, _ = plt.hist(clean_data, bins=get_ideal_number_of_bins(data))
        relative_height_difference = (max(y) - min(y[y != 0])) / max(y)
        self.assertTrue(relative_height_difference < 0.3)

        # normalize over all actions should not have impact as all other actions are the same:
        probabilities_all_dimensions = calculate_probabilites_per_run(data_loader._dataset.data[0])
        self.assertTrue(np.abs(min(probabilities_all_dimensions) - min(probabilities)) < 1e-6)
        self.assertTrue(np.abs(max(probabilities_all_dimensions) - max(probabilities)) < 1e-6)

        # sampling a large batch should have a low relative height
        clean_data = []
        for batch in data_loader.sample_shuffled_batch(batch_size=100):
            clean_data.extend([float(t) for t in batch.outputs['expert'][:, 5]])
        y, x, _ = plt.hist(clean_data, bins=get_ideal_number_of_bins(clean_data))
        relative_height_difference = (max(y) - min(y[y != 0])) / max(y)
        self.assertTrue(relative_height_difference < 0.3)

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
