import os
import unittest

import torch
import numpy as np

from src.core.utils import camelcase_to_snake_format, get_to_root_dir, get_data_dir, safe_wait_till_true


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
        del os.environ['CODEDIR']
        os.chdir('../..')
        self.assertRaises(
            FileNotFoundError,
            get_to_root_dir
        )

    def test_get_data_dir(self):
        # with datadir environment variable
        if "DATADIR" not in os.environ.keys():
            os.environ["DATADIR"] = '/my/wonderful/data/dir'
        self.assertTrue("DATADIR" in os.environ.keys())
        result = get_data_dir(os.environ['HOME'])
        self.assertTrue(result, os.environ["DATADIR"])
        del os.environ['DATADIR']
        self.assertFalse("DATADIR" in os.environ.keys())
        result = get_data_dir(os.environ['HOME'])
        self.assertTrue(result, os.environ["HOME"])

    def test_safe_wait_till_true(self):
        class FakeObject:
            def __init__(self):
                self.field_a = 1
        f = FakeObject()
        safe_wait_till_true('kwargs["f"].field_a', 1, 2, 0.1, f=f)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
