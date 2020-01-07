import unittest


class SingularitySanityCheck(unittest.TestCase):

    def test_cuda_pytorch(self):
        import torch
        self.assertTrue(torch.cuda.is_available())

    def test_gzserver(self):
        import coroutines
        coroutine



if __name__ == '__main__':
    unittest.main()
