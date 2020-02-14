from typing import Union

import numpy as np


class NoiseBase:

    def __init__(self, dimension: tuple = (1,)):
        self._dimension = dimension
        self._noise_value = np.zeros(dimension)

    def sample(self) -> np.ndarray:
        self._noise_value = np.random.normal(size=self._dimension)
        return self._noise_value

    def reset(self) -> None:
        self._noise_value = np.zeros(self._dimension)


class GaussianNoise(NoiseBase):

    def __init__(self, dimension: tuple = (1,),
                 mean: Union[float, np.ndarray] = 0,
                 std: Union[float, np.ndarray] = 1):
        super().__init__(dimension=dimension)
        self._mean = mean
        self._std = std

    def sample(self) -> np.ndarray:
        self._noise_value = np.random.normal(loc=self._mean, scale=self._std, size=self._dimension)
        return self._noise_value


class UniformNoise(NoiseBase):

    def __init__(self, dimension: tuple = (1,),
                 low: Union[float, np.ndarray] = 0,
                 high: Union[float, np.ndarray] = 1):
        super().__init__(dimension=dimension)
        self._low = low
        self._high = high

    def sample(self) -> np.ndarray:
        self._noise_value = np.random.uniform(low=self._low, high=self._high, size=self._dimension)
        return self._noise_value


class OUNoise(NoiseBase):

    def __init__(self, dimension: tuple = (1,),
                 mean: Union[float, np.ndarray] = 0,
                 std: Union[float, np.ndarray] = 1,
                 pullback: Union[float, np.ndarray] = 0.15):
        super().__init__(dimension=dimension)
        self._mean = mean * np.ones(self._dimension) if isinstance(mean, float) else mean
        self._std = std * np.ones(self._dimension) if isinstance(std, float) else std
        self._pullback = pullback * np.ones(self._dimension) if isinstance(pullback, float) else pullback

    def reset(self):
        self._noise_value = np.ones(self._dimension) * self._mean

    def sample(self):
        x = self._noise_value
        dx = self._pullback * (self._mean - x) + self._std * np.random.randn(self._dimension)
        self._noise_value = x + dx
        return self._noise_value
