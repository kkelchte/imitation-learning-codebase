from typing import Union

import numpy as np
import torch


class RunningStatistic(object):
    """Taken from https://github.com/MadryLab/implementation-matters
        Adjusted variance property from self._counter - 1 --> self._counter
    """
    def __init__(self):
        self._counter = 0
        self._mean = None
        self._unnormalized_var = 0

    def add(self, x):
        x = np.asarray(x)
        self._counter += 1
        if self._counter == 1:
            self._mean = x.copy()
        else:
            old_mean = self._mean.copy()
            self._mean = old_mean + (x - old_mean) / self._counter
            if self._unnormalized_var is None:
                self._unnormalized_var = (x - old_mean) * (x - self._mean)
            else:
                self._unnormalized_var = self._unnormalized_var + (x - old_mean) * (x - self._mean)

    def reset(self):
        self._counter = 0
        self._mean = None
        self._unnormalized_var = 0

    @property
    def variance(self):
        return self._unnormalized_var / self._counter if self._counter > 1 else np.square(self._mean)

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return np.sqrt(self.variance)

    @property
    def shape(self):
        return self._mean.shape


class NormalizationFilter:
    """y = (x-mean)/std"""

    def __init__(self, clip: float = -1):
        self._statistic = RunningStatistic()
        self._clip = clip

    def __call__(self, x: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        x = np.asarray(x, dtype=np.float64)
        self._statistic.add(x.copy())
        x -= self._statistic.mean
        x /= (self._statistic.std + 1e-8)
        if self._clip != -1:
            x = np.clip(x, -self._clip, self._clip)
        return x

    def reset(self):
        pass
#        self._statistic.reset()


class ReturnFilter:
    """y = reward / std(returns)"""

    def __init__(self, clip: float = -1, discount: float = 0.99):
        self._retrn = 0
        self._discount = discount
        self._statistic = RunningStatistic()
        self._clip = clip

    def __call__(self, r: float) -> float:
        self._retrn = self._discount * self._retrn + r
        self._statistic.add(self._retrn)
        r /= (self._statistic.std + 1e-8)
        if self._clip != -1:
            r = np.clip(r, -self._clip, self._clip).item()
        return r

    def reset(self):
        self._retrn = 0
        # self._statistic.reset()
