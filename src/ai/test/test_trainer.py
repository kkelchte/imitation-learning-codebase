import unittest
import os
import shutil

import numpy as np

from src.ai.evaluator import Evaluator, EvaluatorConfig
from src.ai.model import Model, ModelConfig
from src.ai.trainer import TrainerConfig, Trainer
from src.core.utils import get_filename_without_extension, get_date_time_tag


class TestTrainer(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        architecture = 'tiny_net_v0'
        model_config = {'output_path': self.output_dir,
                        'load_checkpoint_dir': None,
                        'architecture': architecture,
                        'output_sizes': [[6]],
                        'dropout': 0.}
        self.model = Model(config=ModelConfig().create(config_dict=model_config))

    def test_evaluate_model(self):
        evaluator_config_dict = {
            'output_path': os.path.join(self.output_dir, 'models',
                                        f'{get_date_time_tag()}_{self.model._config.architecture}'),
            'data_loader_config': {
                'output_path': '/esat/opal/kkelchte/experimental_data/dummy_dataset',
                'data_directories': ['raw_data/20-02-06_13-32-24',
                                     'raw_data/20-02-06_13-32-43'],
                'inputs': ['forward_camera'],
                'outputs': ['ros_expert']},
            'device': 'cpu',
            'criterion': 'MSELoss',
        }
        evaluator = Evaluator(config=EvaluatorConfig().create(config_dict=evaluator_config_dict), model=self.model)
        error = evaluator.evaluate()
        self.assertTrue(not np.isnan(error))

    def test_train_model(self):
        trainer_config_dict = {
            'output_path': os.path.join(self.output_dir, 'models',
                                        f'{get_date_time_tag()}_{self.model._config.architecture}'),
            'data_loader_config': {
                'output_path': '/esat/opal/kkelchte/experimental_data/dummy_dataset',
                'data_directories': ['raw_data/20-02-06_13-32-24'],
                'inputs': ['forward_camera'],
                'outputs': ['ros_expert']},
            'device':  'cuda',
            'criterion':  'MSELoss',
            'batch_size':  32,
            'optimizer':  'SGD',
            'learning_rate':  0.0001,
        }
        trainer = Trainer(config=TrainerConfig().create(config_dict=trainer_config_dict), model=self.model)
        initial_error = trainer.train()
        self.assertTrue(not np.isnan(initial_error))
        for e in range(10):
            error = trainer.train()
        self.assertTrue(error < initial_error)

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
