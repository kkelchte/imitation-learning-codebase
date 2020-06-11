import os
import unittest

import torch
import numpy as np

from src.core.utils import camelcase_to_snake_format, get_to_root_dir, select


class TestUtils(unittest.TestCase):

    def test_camelcase_to_snakeformat(self):
        self.assertEqual(camelcase_to_snake_format('ThisIsAFirstTest'),
                         'this_is_a_first_test')
        self.assertEqual(camelcase_to_snake_format('ThisIsA--SecondTest'),
                         'this_is_a--second_test')

    def test_get_to_root_dir(self):
        # normal usage:
        get_to_root_dir()
        self.assertTrue('ROOTDIR' in os.listdir())
        # change to a code directory
        os.chdir('src/core/test')
        get_to_root_dir()
        self.assertTrue('ROOTDIR' in os.listdir())
        # check raise error
        os.chdir('../..')
        self.assertRaises(
            FileNotFoundError,
            get_to_root_dir
        )

    def test_select(self):
        data = list(range(10))
        indices = [3, 5]
        result = select(data, indices)
        self.assertEqual(result, [3, 5])

        data = torch.as_tensor([[v, 1, 10 - v] for v in range(10)])
        indices = [3, 5]
        result = select(data, indices)
        self.assertEqual((result - torch.as_tensor([[3, 1, 7], [5, 1, 5]])).sum().item(), 0)

        data = np.asarray([[v, 1, 10 - v] for v in range(10)])
        indices = [3, 5]
        result = select(data, indices)
        self.assertEqual((result - np.asarray([[3, 1, 7], [5, 1, 5]])).sum(), 0)


if __name__ == '__main__':
    unittest.main()
