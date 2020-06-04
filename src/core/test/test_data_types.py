import unittest

import torch

from src.core.data_types import Distribution, Dataset, Experience


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

    def test_dataset_subsample(self):
        run_length = 10
        subsample = 3
        dataset = Dataset()
        for run_index in range(3):
            for step_index in range(run_length + run_index):
                dataset.append(Experience(
                    observation=torch.as_tensor((step_index,)),
                    action=torch.as_tensor((0,)),
                    reward=torch.as_tensor((0,)),
                    done=torch.as_tensor((0,)) if step_index != run_length + run_index - 1 else torch.as_tensor((1,))
                ))
        dataset.subsample(subsample)
        for exp_index in range(len(dataset)):
            self.assertTrue(dataset.observations[exp_index].item() % subsample == 0
                            or dataset.done[exp_index].item() == 1)


if __name__ == '__main__':
    unittest.main()
