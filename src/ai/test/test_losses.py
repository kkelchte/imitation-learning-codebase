import os
import shutil
import unittest

import torch

from src.ai.evaluator import Evaluator, EvaluatorConfig
from src.ai.losses import *
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

    def test_coral_loss(self):
        loss = Coral()
        batch_size = 100000
        feature_size = (256,)
        source = torch.randn(size=(batch_size, *feature_size))
        target = torch.randn(size=(batch_size, *feature_size))
        coral = loss(source, target)
        self.assertTrue(coral < 3)

        scale = 10
        source = scale * torch.randn(size=(batch_size, *feature_size))
        target = torch.randn(size=(batch_size, *feature_size))
        coral = loss(source, target)
        self.assertTrue(1000 < coral < 2000)

    def test_mmd_loss(self):
        batch_size = 10
        feature_size = (64, 25, 25)
        source = 10 * torch.randn(size=(batch_size, *feature_size), requires_grad=True) + 10
        target = torch.randn(size=(batch_size, *feature_size), requires_grad=True)

        mmd_loss_zhao = MMDLossZhao(kernel_mul=1.0, kernel_num=2, fix_sigma=None)
        print(mmd_loss_zhao(source, target))

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
