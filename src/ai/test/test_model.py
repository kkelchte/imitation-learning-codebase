import os
import shutil
import unittest
from typing import List

from src.ai.model import Model, ModelConfig
from src.core.utils import get_filename_without_extension
from src.data.dataset_loader import DataLoader, DataLoaderConfig


class TestModel(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'test_dir/{get_filename_without_extension(__file__)}'
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

    def _test_input_output_of_architecture(self, architecture: str,
                                           input_signals: List[str],
                                           output_signals: List[str]):
        model_config = {'output_path': self.output_dir,
                        'load_checkpoint_path': None,
                        'architecture': architecture,
                        'dropout': 0.}
        model = Model(config=ModelConfig().create(config_dict=model_config))

        data_loader_config = {
            'output_path': '/esat/opal/kkelchte/experimental_data/dummy_dataset',
            'data_directories': ['raw_data/20-02-06_13-32-24'],
            'inputs': input_signals,
            'outputs': output_signals,
        }
        data_loader = DataLoader(config=DataLoaderConfig().create(config_dict=data_loader_config))
        data_loader.load_dataset(model.get_input_sizes(), model.get_output_sizes())
        for batch in data_loader.sample_shuffled_batch(batch_size=5, max_number_of_batches=1):
            model_outputs = model.forward(batch.get_input())
            self.assertTrue(len(model_outputs) == len(batch.get_output()))
            for o_i, o in model_outputs:
                self.assertTrue(o.size() == batch.get_output()[o_i].size())

    def test_input_output(self) -> None:
        self._test_input_output_of_architecture(architecture='tiny_net_v0',
                                                input_signals=['forward_camera'],
                                                output_signals=['ros_expert'])
        # For each architecture: input-output test should be added.

    def test_save_load_checkpoint(self) -> None:
        architecture = 'tiny_net_v0'
        model_config = {'output_path': self.output_dir,
                        'load_checkpoint_path': None,
                        'architecture': architecture,
                        'dropout': 0.}
        model = Model(config=ModelConfig().create(config_dict=model_config))
        validation_param = model.get_parameters()[0]
        model.save_to_checkpoint(tag='5')
        self.assertTrue(os.path.isfile(os.path.join(self.output_dir, 'torch_checkpoints', 'checkpoint_latest')))
        self.assertTrue(os.path.isfile(os.path.join(self.output_dir, 'torch_checkpoints', 'checkpoint_5')))

        model_config['load_checkpoint_path'] = os.path.join(self.output_dir, 'torch_checkpoints', 'checkpoint_latest')
        new_model = Model(config=ModelConfig().create(config_dict=model_config))
        self.assertTrue(validation_param[:] == new_model.get_parameters()[0][:])

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
