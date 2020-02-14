import unittest

from src.sim.common.noise import *


class TestNoise(unittest.TestCase):

    def test_normal_noise(self):
        noise = NoiseBase()
        data = []
        for i in range(1000):
            data.append(float(noise.sample()))
        self.assertTrue(np.mean(data) < 1e-6)

    def test_gaussian_noise(self):
        noise = GaussianNoise(mean=134,
                              std=10,
                              dimension=(128, 128, 3))
        data = []
        for i in range(10):
            data.append(noise.sample())
        self.assertTrue(np.mean(np.asarray(data)) - 134 < 1e-6)

    def test_uniform_noise(self):
        noise = UniformNoise(low=0,
                             high=255,
                             dimension=(128, 128, 3))
        data = []
        for i in range(10):
            data.append(noise.sample())
        self.assertTrue(np.amin(np.asarray(data)) >= 0)
        self.assertTrue(np.amax(np.asarray(data)) <= 255)

    def test_ou_noise(self):
        noise = OUNoise(mean=134,
                        std=10,
                        pullback=0.15,
                        dimension=(6, 1))
        data = []
        for i in range(100):
            data.append(noise.sample())
        print(data)
        self.assertTrue(np.amin(np.asarray(data)) >= 0)
        self.assertTrue(np.amax(np.asarray(data)) <= 255)


if __name__ == '__main__':
    unittest.main()
