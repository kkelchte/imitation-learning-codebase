import unittest

import torch

from src.core.data_types import Distribution


class TestDatatypes(unittest.TestCase):

    def test_distribution_list(self):
        data = [10] * 5
        distribution = Distribution(data)
        self.assertEqual(distribution.max, 10)
        self.assertEqual(distribution.min, 10)
        self.assertEqual(distribution.mean, 10)
        self.assertEqual(distribution.std, 0)

    def test_distribution_tensor(self):
        data = [10] * 5
        distribution = Distribution(torch.as_tensor(data))
        self.assertEqual(distribution.max, 10)
        self.assertEqual(distribution.min, 10)
        self.assertEqual(distribution.mean, 10)
        self.assertEqual(distribution.std, 0)


if __name__ == '__main__':
    unittest.main()
