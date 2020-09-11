import sys
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

    def test_dataset_size(self):
        dataset = Dataset()
        dataset.append(Experience(observation=torch.as_tensor([0]*10),
                                  action=torch.as_tensor([1]*3),
                                  reward=torch.as_tensor(0),
                                  done=torch.as_tensor(2)))
        first_size = dataset.get_memory_size()
        dataset.append(Experience(observation=torch.as_tensor([0]*10),
                                  action=torch.as_tensor([1]*3),
                                  reward=torch.as_tensor(0),
                                  done=torch.as_tensor(2)))
        self.assertEqual(2 * first_size, dataset.get_memory_size())
        dataset = Dataset()
        dataset.append(Experience(observation=torch.as_tensor([0] * 10, dtype=torch.float32),
                                  action=torch.as_tensor([1] * 3, dtype=torch.float32),
                                  reward=torch.as_tensor(0, dtype=torch.float32),
                                  done=torch.as_tensor(2, dtype=torch.float32)))
        second_size = dataset.get_memory_size()
        self.assertEqual(first_size, 2*second_size)


if __name__ == '__main__':
    unittest.main()
