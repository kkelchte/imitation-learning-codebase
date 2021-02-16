import os
import shutil
import unittest

import numpy as np
import torch

from src.ai.base_net import ArchitectureConfig
from src.ai.evaluator import Evaluator, EvaluatorConfig
from src.ai.utils import generate_random_dataset_in_raw_data, DiscreteActionMapper, \
    clip_action_according_to_playfield_size
from src.core.utils import get_to_root_dir, get_filename_without_extension
from src.ai.architectures import *  # Do not remove
from src.data.data_loader import DataLoaderConfig, DataLoader


class UtilsTest(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

    def test_discrete_action_mapper(self):
        action_values = [
            torch.as_tensor([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
            torch.as_tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
            torch.as_tensor([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
        ]
        discrete_action_mapper = DiscreteActionMapper(action_values)
        actions = [torch.as_tensor([0, 0, 0, 0, 0, -1])] * 10
        indices = [discrete_action_mapper.tensor_to_index(a) for a in actions]
        [self.assertEqual(i, 0) for i in indices]
        self.assertLess((discrete_action_mapper.index_to_tensor(1) -
                         torch.as_tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])).sum(),
                        0.00001)

    def test_clip_action_according_to_playfield_size(self):
        playfield_size = (1., 1., 1.)
        # default usage
        inputs = np.asarray([0, 0, 0,
                             0, 0, 0,
                             0, 3, 0])
        actions = np.asarray([4, 0, 0,
                              0, 5, 0,
                              0, 1])
        result = clip_action_according_to_playfield_size(inputs, actions, playfield_size=playfield_size)
        self.assertListEqual(list(result), list(actions))
        # clip all actions one time in + and -
        offset = 0.1
        for agent in ['tracking', 'fleeing']:
            for direction in range(3):
                for sign in [+1, -1]:
                    inputs = np.zeros((9,))
                    index = direction if agent == 'tracking' else direction + 3
                    inputs[index] = sign * (playfield_size[direction] + offset)
                    actions = np.arange(1, 9)
                    print(f'inputs: {inputs}')
                    print(f'actions: {actions}')
                    result = clip_action_according_to_playfield_size(inputs, actions, playfield_size=playfield_size)
                    self.assertEqual(result[index], 0 if sign > 0 else index + 1)
                    print(f'result: {result}')

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
