import os
import shutil
import unittest

import torch
import numpy as np

from src.core.tensorboard_wrapper import TensorboardWrapper
from src.core.utils import camelcase_to_snake_format, get_to_root_dir, get_filename_without_extension


class TestTensorboard(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

    def test_video(self):
        frames = []
        for _ in range(100):
            frames.append(np.random.randint(0, 255, (60, 60), dtype=np.uint8))
        writer = TensorboardWrapper(log_dir=self.output_dir)
        writer.write_gif(frames)
        writer.close()

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
