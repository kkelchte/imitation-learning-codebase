import unittest

import numpy as np

from src.core.filters import RunningStatistic, NormalizationFilter, ReturnFilter
from src.core.utils import get_to_root_dir


class TestFilters(unittest.TestCase):

    def test_running_statistic(self):
        statistic = RunningStatistic()
        for _ in range(10):
            statistic.add(_)
        self.assertEqual(np.mean(list(range(10))), statistic.mean)
        self.assertEqual(np.std(list(range(10))), statistic.std)

    def test_filter(self):
        filter_to_test = NormalizationFilter()
        solutions = []
        for x in range(100):
            solutions.append(filter_to_test(np.asarray([0,
                                                        1,
                                                        0 if x < 5 else 1,
                                                        -1 if x % 2 == 0 else 1])))
        self.assertEqual(solutions[-1][0], 0)
        self.assertEqual(solutions[-1][1], 0)
        self.assertLess(abs(filter_to_test._statistic.std[2] - np.std([0 if _ < 5 else 1 for _ in range(100)])), 1e-10)
        self.assertLess(abs(filter_to_test._statistic.mean[3]), 1e-10)

        filter_to_test.reset()
        self.assertEqual(filter_to_test._statistic._counter, 0)
        for _ in range(10):
            filter_to_test(_)
        self.assertEqual(np.mean(list(range(10))), filter_to_test._statistic.mean)
        self.assertEqual(np.std(list(range(10))), filter_to_test._statistic.std)

    def test_return_filter(self):
        filter_to_test = ReturnFilter(clip=5)

        print(f'\nconstant 10x3:')
        for _ in range(10):
            print(filter_to_test(3))
        filter_to_test.reset()

        print(f'\nconstant 10x 100:')
        for _ in range(10):
            print(filter_to_test(100))
        filter_to_test.reset()

        print(f'\nincrement range(10):')
        for _ in range(10):
            print(filter_to_test(_))
        filter_to_test.reset()

        print(f'\n-1 <> +1 @%2:')
        for _ in range(10):
            print(filter_to_test(-1 if _ % 2 == 0 else 1))
        filter_to_test.reset()

        print(f'\nblok 5x0 -> 15x1:')
        for _ in range(15):
            print(filter_to_test(0 if _ < 5 else 1))
        filter_to_test.reset()

        # Conclusion: by dividing the reward through the standard deviation of a running return
        # -> constant (non-zero) rewards decrease gradually in value from 2 --> 0 despite the initial large value
        # -> even incrementing rewards decrease gradually (but slower) as the standard deviation increases
        #    slightly faster than the reward
        # -> strong fluctuations in reward (-1:1) are emphasized (-2:2)
        #    but that again decreased slowly if this continuous
        # -> sudden jumps 0 -> 1 are emphasized (2) but then quickly smoothed if value remains constant


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
