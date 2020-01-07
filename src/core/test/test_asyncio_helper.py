import unittest
import asyncio

from src.core.asyncio_helper import run


class TestAsyncioHelper(unittest.TestCase):

    def test_run(self):
        stdout, stderr = asyncio.run(run('ls'))
        self.assertTrue(stdout)
        self.assertFalse(stderr)


if __name__ == '__main__':
    unittest.main()
