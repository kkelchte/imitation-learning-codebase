import os
import shutil
import unittest

import numpy as np

from src.core.utils import get_filename_without_extension, get_data_dir
from src.sim.ros.src.utils import calculate_bounding_box

""" Test Utils
"""


class TestUtils(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'

    def test_bounding_box(self):
        resolution = (100, 100)
        tracking_agent_position = [0, 0, 1]
        tracking_agent_orientation = [0, 0, 0]
        fleeing_agent_position = [1, 1, 1]
        position, width, height = calculate_bounding_box(state=[*tracking_agent_position,
                                                                *fleeing_agent_position,
                                                                *tracking_agent_orientation],
                                                         resolution=resolution)
        import matplotlib.pyplot as plt
        frame = np.zeros(resolution)
        frame[position:position + width,
              position:position + height] = 1
        plt.imshow(frame)
        plt.show()

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
