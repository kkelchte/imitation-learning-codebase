import os
import unittest

import torch
import numpy as np

from src.core.utils import camelcase_to_snake_format, get_to_root_dir


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


if __name__ == '__main__':
    unittest.main()
