from typing import Union

import numpy as np
import torch


class RunningStatistic(object):
    """Taken from https://github.com/MadryLab/implementation-matters
    """
    def __init__(self):
        self._counter = 0
        self._mean = None
        self._unnormalized_var = None

    def add(self, x):
        x = np.asarray(x)
        self._counter += 1
        if self._counter == 1:
            self._mean = x
        else:
            old_mean = self._mean.copy()
            self._mean[...] = old_mean + (x - old_mean) / self._counter
            self._unnormalized_var = self._unnormalized_var + (x - old_mean) * (x - self._mean) \
                if self._unnormalized_var is not None else (x - old_mean) * (x - self._mean)

    @property
    def variance(self):
        if self._counter > 1:
            return self._unnormalized_var / (self._counter - 1)
        elif self._counter == 1:
            return np.square(self._mean)
        else:
            return None

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return np.sqrt(self.variance)

    @property
    def shape(self):
        return self._mean.shape

    def get_checkpoint(self) -> dict:
        """
        :return: a dictionary with the necessary fields to store current filters state.
        """
        return {
            'counter': self._counter,
            'mean': self._mean,
            'unnormalized_var': self._unnormalized_var
        }

    def load_checkpoint(self, checkpoint) -> None:
        """
        Load checkpoint for running statistic.
        :param checkpoint: dictionary
        :return: None
        """
        self._counter = checkpoint['counter']
        self._mean = checkpoint['mean']
        self._unnormalized_var = checkpoint['unnormalized_var']


class NormalizationFilter:
    """
    Normalize data stream with (x-average)/std with average and std provided or estimated from a running statistic.
    """

    def __init__(self, clip: float = -1):
        """
        :param clip: clip output of filter
        :param mean: average. If provided, use this value instead of the running statistic estimate.
        :param std: standard deviation. If provided, use this value instead of the running statistic estimate.
        """
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

    def get_checkpoint(self) -> dict:
        """
        :return: a dictionary with the necessary fields to store current filters state.
        """
        return {
            'running_statistic': self._statistic.get_checkpoint(),
        }

    def load_checkpoint(self, checkpoint) -> None:
        """
        Load checkpoint for running statistic.
        :param checkpoint: dictionary
        :return: None
        """
        self._statistic.load_checkpoint(checkpoint['running_statistic'])


class ReturnFilter(NormalizationFilter):
    """y = reward / std(returns)"""

    def __init__(self, clip: float = -1, discount: float = 0.99, std=None):
        """
        :param clip: clip output of filter
        :param discount: discount accumulated rewards.
        :param std: standard deviation. If provided, use this value instead of the running statistic estimate.
        """
        super().__init__(clip)
        self._return = None
        self._discount = discount
        self._std = std

    def __call__(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        self._return = self._discount * self._return + r if self._return is not None else r
        self._statistic.add(self._return)
        r /= (self._statistic.std + 1e-8) if self._std is None else self._std
        if self._clip != -1:
            r = np.clip(r, -self._clip, self._clip)
        return r

    def reset(self):
        self._return = None

    def get_checkpoint(self) -> dict:
        """
        :return: a dictionary with the necessary fields to store current filters state.
        """
        checkpoint = super().get_checkpoint()
        checkpoint['return'] = self._return
        return checkpoint

    def load_checkpoint(self, checkpoint) -> None:
        """
        Load checkpoint for running statistic.
        :param checkpoint: dictionary
        :return: None
        """
        super().load_checkpoint(checkpoint)
        self._return = checkpoint['return']
