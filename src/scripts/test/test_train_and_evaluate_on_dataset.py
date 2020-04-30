import os
import shutil
import unittest
from copy import deepcopy
from glob import glob

import numpy as np

from src.ai.base_net import InitializationType, ArchitectureConfig
from src.ai.evaluator import Evaluator, EvaluatorConfig
from src.ai.trainer import TrainerConfig, Trainer
from src.ai.utils import generate_random_dataset_in_raw_data
from src.core.utils import get_to_root_dir, get_filename_without_extension
from src.ai.architectures import *  # Do not remove
from src.data.data_loader import DataLoaderConfig, DataLoader
from src.scripts.experiment import Experiment, ExperimentConfig

experiment_config = {
    "output_path": "/tmp",
    "number_of_epochs": 4,
    "architecture_config": {
        "architecture": "tiny_128_rgb_1c",
        "load_checkpoint_dir": None,
        "initialisation_type": InitializationType.Xavier,
        "initialisation_seed": 0,
        "device": 'cpu'},
    "trainer_config": {
        "save_checkpoint_every_n": 2,
        "data_loader_config": {"batch_size": 5,
                               "hdf5_file": "train.hdf5"},
        "criterion": "MSELoss",
        "device": "cpu"},
    "evaluator_config": {
        "data_loader_config": {"batch_size": 1,
                               "hdf5_file": "validation.hdf5"},
        "criterion": "MSELoss",
        "device": "cpu"},
}


class DatasetExperimentsTest(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        experiment_config['output_path'] = self.output_dir
        experiment_config['architecture_config']['output_path'] = self.output_dir

    def test_train_model_on_generated_dataset(self):
        network = eval(experiment_config['architecture_config']['architecture']).Net(
            config=ArchitectureConfig().create(config_dict=experiment_config['architecture_config'])
        )
        info = generate_random_dataset_in_raw_data(output_dir=self.output_dir,
                                                   num_runs=5,
                                                   input_size=network.input_size,
                                                   output_size=network.output_size,
                                                   continuous=network.continuous_output,
                                                   fixed_output_value=0,
                                                   store_hdf5=True)
        experiment = Experiment(config=ExperimentConfig().create(config_dict=experiment_config))
        experiment.run()

        # check if 5 + 2 checkpoints were stored in torch_checkpoints
        self.assertEqual(len([f for f in os.listdir(os.path.join(self.output_dir, 'torch_checkpoints'))
                              if f.endswith('ckpt')]), 4)

    def test_train_model_on_generated_dataset_with_tensorboard(self):
        network = eval(experiment_config['architecture_config']['architecture']).Net(
            config=ArchitectureConfig().create(config_dict=experiment_config['architecture_config'])
        )
        info = generate_random_dataset_in_raw_data(output_dir=self.output_dir,
                                                   num_runs=5,
                                                   input_size=network.input_size,
                                                   output_size=network.output_size,
                                                   continuous=network.continuous_output,
                                                   fixed_output_value=0,
                                                   store_hdf5=True)
        experiment_config['tensorboard'] = True
        experiment = Experiment(config=ExperimentConfig().create(config_dict=experiment_config))
        experiment.run()
        self.assertGreater(len(glob(os.path.join(self.output_dir, 'events.*'))), 0)

    def test_train_model_on_external_dataset_as_hdf5(self):
        network = eval(experiment_config['architecture_config']['architecture']).Net(
            config=ArchitectureConfig().create(config_dict=experiment_config['architecture_config'])
        )
        external_dataset = f'{os.environ["PWD"]}/test_dir/external_dataset'
        os.makedirs(external_dataset, exist_ok=True)
        info = generate_random_dataset_in_raw_data(output_dir=external_dataset,
                                                   num_runs=5,
                                                   input_size=network.input_size,
                                                   output_size=network.output_size,
                                                   continuous=network.continuous_output,
                                                   fixed_output_value=0,
                                                   store_hdf5=True)
        self.assertTrue(os.path.isfile(os.path.join(external_dataset, 'train.hdf5')))
        self.assertTrue(os.path.isfile(os.path.join(external_dataset, 'validation.hdf5')))

        experiment_config["trainer_config"]["data_loader_config"]["hdf5_file"] = os.path.join(external_dataset,
                                                                                              'train.hdf5')
        experiment_config["evaluator_config"]["data_loader_config"]["hdf5_file"] = os.path.join(external_dataset,
                                                                                                'validation.hdf5')
        experiment = Experiment(config=ExperimentConfig().create(config_dict=experiment_config))
        experiment.run()

        # check if 5 + 2 checkpoints were stored in torch_checkpoints
        self.assertTrue(len([f for f in os.listdir(os.path.join(self.output_dir, 'torch_checkpoints'))
                             if f.endswith('ckpt')]), 4)
        shutil.rmtree(external_dataset, ignore_errors=True)

    def test_train_model_on_external_dataset_as_raw_data(self):
        network = eval(experiment_config['architecture_config']['architecture']).Net(
            config=ArchitectureConfig().create(config_dict=experiment_config['architecture_config'])
        )
        external_dataset = f'{os.environ["PWD"]}/test_dir/external_dataset'
        os.makedirs(external_dataset, exist_ok=True)
        info = generate_random_dataset_in_raw_data(output_dir=external_dataset,
                                                   num_runs=5,
                                                   input_size=network.input_size,
                                                   output_size=network.output_size,
                                                   continuous=network.continuous_output,
                                                   fixed_output_value=0,
                                                   store_hdf5=False)

        raw_data_directories = [
            os.path.join(external_dataset, 'raw_data', d)
            for d in os.listdir(os.path.join(external_dataset, 'raw_data'))
        ]
        experiment_config["trainer_config"]["data_loader_config"]["hdf5_file"] = ""
        experiment_config["evaluator_config"]["data_loader_config"]["hdf5_file"] = ""
        experiment_config["trainer_config"]["data_loader_config"]["data_directories"] = raw_data_directories[:-2]
        experiment_config["evaluator_config"]["data_loader_config"]["data_directories"] = raw_data_directories[-2:]
        experiment = Experiment(config=ExperimentConfig().create(config_dict=experiment_config))
        experiment.run()

        # check if 5 + 2 checkpoints were stored in torch_checkpoints
        self.assertTrue(len([f for f in os.listdir(os.path.join(self.output_dir, 'torch_checkpoints'))
                             if f.endswith('ckpt')]), 4)
        shutil.rmtree(external_dataset, ignore_errors=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
