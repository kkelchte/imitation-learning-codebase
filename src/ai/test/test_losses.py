import os
import shutil
import unittest

import torch

from src.ai.evaluator import Evaluator, EvaluatorConfig
from src.ai.losses import WeightedBinaryCrossEntropyLoss
from src.core.utils import get_to_root_dir, get_filename_without_extension


class UtilsTest(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

    def test_WeightedBinaryCrossEntropyLoss(self):
        batch_size = 10
        input_size = (32, 32)
        inputs = torch.randint(low=0, high=100, size=(batch_size, *input_size)).float() / 100
        targets = ((torch.randn(size=(batch_size, *input_size)).sign() + 1) / 2).float()
        loss = WeightedBinaryCrossEntropyLoss(beta=0.9)
        result = loss(inputs, targets)
        self.assertTrue(0 < result)
        self.assertTrue(torch.isnan(result).sum().item() == 0)

    def test_loading_weighted_binary_loss_from_evaluator(self):
        evaluator_base_config = {
            "output_path": self.output_dir,
            "data_loader_config": {},
            "criterion": "WeightedBinaryCrossEntropyLoss",
            "criterion_args_str": 'beta=0.9',
            "device": "cpu"
        }
        evaluator = Evaluator(config=EvaluatorConfig().create(config_dict=evaluator_base_config), network=None)
        self.assertTrue(isinstance(evaluator._criterion, WeightedBinaryCrossEntropyLoss))

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
