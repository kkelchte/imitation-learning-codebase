import unittest
import asyncio

from src.core.asyncio_helper import run


class SingularitySanityCheck(unittest.TestCase):

    def test_cuda_pytorch(self):
        import torch
        self.assertTrue(torch.cuda.is_available())

    def test_gzserver(self):
        print('Open new terminal and run: \n'
              'singularity run --nv ros-gazebo-cuda_0.0.2.sif gzclient')
        stdout, stderr = asyncio.run(run('gzserver --verbose &'))
        self.assertFalse(stderr)


if __name__ == '__main__':
    unittest.main()
