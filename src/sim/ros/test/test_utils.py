import os
import shutil
import unittest

import numpy as np
import matplotlib.pyplot as plt

from src.core.utils import get_filename_without_extension, get_data_dir
from src.sim.ros.src.utils import calculate_bounding_box, distance, array_to_combined_global_pose, get_iou, \
    process_combined_global_poses

""" Test Utils
"""


class TestUtils(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'

    def test_distance(self):
        self.assertEqual(distance([0, 0, 2], [0, 0, 0]), 2)
        self.assertEqual(distance([0, 0], [0, 1]), 1)
        distance(np.asarray([0, 0, 2]).reshape((1, 1, 3)), np.asarray([0, 0, 0]).reshape((1, 1, 3)))

    def test_bounding_box(self):
        resolution = (100, 100)
        tracking_agent_position = [0, 0, 1]
        tracking_agent_orientation = [0, 0, 0]
        fleeing_agent_position = [1, 1, 1]
        position, width, height = calculate_bounding_box(state=[*tracking_agent_position,
                                                                *fleeing_agent_position,
                                                                *tracking_agent_orientation],
                                                         resolution=resolution)
        frame = np.zeros(resolution)
        frame[position[0]:position[0]+width,
              position[1]:position[1]+height] = 1
        plt.imshow(frame)
        plt.show()

    def test_intersection_over_union(self):
        tracking_agent_position = [0, 0, 1]
        tracking_agent_orientation = [0, 0, 0]
        fleeing_agent_position = [1, 1, 1]
        info = {'combined_global_poses': array_to_combined_global_pose([*tracking_agent_position,
                                                                        *fleeing_agent_position,
                                                                        *tracking_agent_orientation])}
        result = get_iou(info)
        self.assertEqual(result, 5)  # TODO replace 0 with desired result

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
