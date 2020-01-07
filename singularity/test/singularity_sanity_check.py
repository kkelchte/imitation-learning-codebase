import unittest
import asyncio

from src.core.asyncio_helper import run


class SingularitySanityCheck(unittest.TestCase):

    # def test_cuda_pytorch(self):
    #     import torch
    #     self.assertTrue(torch.cuda.is_available())

    def test_gzserver(self):
        asyncio.run(run('gzserver'))


if __name__ == '__main__':
    unittest.main()
