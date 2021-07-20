import os
import shutil
import unittest

import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import PointStamped, PoseStamped, Pose, Point, Quaternion
from scipy.spatial.transform import Rotation as R

from math import pi
from src.core.utils import get_filename_without_extension, get_data_dir
from src.sim.ros.src.utils import calculate_bounding_box, distance, array_to_combined_global_pose, get_iou, \
    process_combined_global_poses, transform, to_ros_time, calculate_relative_orientation

""" Test Utils
"""


class TestUtils(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'

    def test_calculate_relative_orientation(self):
        # Looking in +y direction (90° ccw)
        robot_pose = PoseStamped(pose=Pose(position=Point(x=1, y=0, z=0),
                                           orientation=Quaternion(x=0, y=0, z=0.7071068, w=0.7071068)))
        # reference (0, 1) should result in +pi/4
        reference_pose = PointStamped(point=Point(x=0, y=1, z=0))
        result = calculate_relative_orientation(robot_pose, reference_pose)
        self.assertAlmostEqual(result, np.pi/4, places=3)
        # reference (0, -1) should result in +3pi/4
        reference_pose = PointStamped(point=Point(x=0, y=-1, z=0))
        result = calculate_relative_orientation(robot_pose, reference_pose)
        self.assertAlmostEqual(result, np.pi * 3 / 4, places=3)
        # reference (2, -1) should result in -3pi/4
        reference_pose = PointStamped(point=Point(x=2, y=-1, z=0))
        result = calculate_relative_orientation(robot_pose, reference_pose)
        self.assertAlmostEqual(result, -np.pi * 3 / 4 + 2*np.pi, places=3)
        # reference (2, 1) should result in -pi/4
        reference_pose = PointStamped(point=Point(x=2, y=1, z=0))
        result = calculate_relative_orientation(robot_pose, reference_pose)
        self.assertAlmostEqual(result, -np.pi / 4, places=3)

    def test_to_ros_time(self):
        result = to_ros_time(5)
        self.assertEqual(result.secs, 5)
        self.assertEqual(result.nsecs, 0)

        result = to_ros_time(5.1)
        self.assertEqual(result.secs, 5)
        self.assertLessEqual(abs(result.nsecs - 10**8), 1)

    def test_distance(self):
        self.assertEqual(distance([0, 0, 2], [0, 0, 0]), 2)
        self.assertEqual(distance([0, 0], [0, 1]), 1)
        distance(np.asarray([0, 0, 2]).reshape((1, 1, 3)), np.asarray([0, 0, 0]).reshape((1, 1, 3)))

    def test_bounding_box(self):
        resolution = (400, 400)
        tracking_agent_position = [0, 0, 4]
        tracking_agent_orientation = [0, 0, 0]
        fleeing_agent_position = [2, 0, 1]
        bounding_boxes = calculate_bounding_box(state=[*tracking_agent_position,
                                                       *fleeing_agent_position,
                                                       *tracking_agent_orientation],
                                                resolution=resolution)
        position = bounding_boxes[3]
        width = bounding_boxes[4]
        height = bounding_boxes[5]
        frame = np.zeros(resolution)
        frame[position[1]-height//2:position[1]+height//2,
              position[0]-width//2:position[0]+width//2] = 1
        plt.imshow(frame)
        plt.show()

        self.assertEqual(bounding_boxes, ((500, 500), 66, 66, (500, 500), 66, 66))

        fleeing_agent_position = [3, 2, 1]
        bounding_boxes = calculate_bounding_box(state=[*tracking_agent_position,
                                                       *fleeing_agent_position,
                                                       *tracking_agent_orientation],
                                                resolution=resolution)
        self.assertEqual(bounding_boxes, ((500, 500), 66, 66, (166, 500), 63, 66))

        fleeing_agent_position = [3, 0, 1]
        bounding_boxes = calculate_bounding_box(state=[*tracking_agent_position,
                                                       *fleeing_agent_position,
                                                       *tracking_agent_orientation],
                                                resolution=resolution)
        self.assertEqual(bounding_boxes, ((500, 500), 66, 66, (833, 500), 63, 66))

        fleeing_agent_position = [4, 1, 1]
        tracking_agent_orientation = [0.3, 0, 0]
        bounding_boxes = calculate_bounding_box(state=[*tracking_agent_position,
                                                       *fleeing_agent_position,
                                                       *tracking_agent_orientation],
                                                resolution=resolution)
        self.assertEqual(bounding_boxes, ((500, 500), 66, 66, (190, 500), 50, 52))
        # position = bounding_boxes[3]
        # width = bounding_boxes[4]
        # height = bounding_boxes[5]
        # frame = np.zeros(resolution)
        # frame[position[1]-height//2:position[1]+height//2,
        #      position[0]-width//2:position[0]+width//2] = 1
        # plt.imshow(frame)
        # plt.show()
        # a=100

    def test_intersection_over_union(self):
        tracking_agent_position = [0, 1, 1]
        tracking_agent_orientation = [0, 0, 0]
        fleeing_agent_position = [3, 1, 1]
        info = {'combined_global_poses': array_to_combined_global_pose([*tracking_agent_position,
                                                                        *fleeing_agent_position,
                                                                        *tracking_agent_orientation])}
        result = get_iou(info)
        self.assertEqual(result, 1)

        fleeing_agent_position = [3, 2, 1]
        info = {'combined_global_poses': array_to_combined_global_pose([*tracking_agent_position,
                                                                        *fleeing_agent_position,
                                                                        *tracking_agent_orientation])}
        result = get_iou(info)
        self.assertEqual(result, 0)

        fleeing_agent_position = [3, 0.9, 1.1]
        info = {'combined_global_poses': array_to_combined_global_pose([*tracking_agent_position,
                                                                        *fleeing_agent_position,
                                                                        *tracking_agent_orientation])}
        result = get_iou(info)
        self.assertEqual(round(result, 3), 0.143)

        fleeing_agent_position = [3, 0.99, 1.01]
        info = {'combined_global_poses': array_to_combined_global_pose([*tracking_agent_position,
                                                                        *fleeing_agent_position,
                                                                        *tracking_agent_orientation])}
        result = get_iou(info)
        pos0, w0, h0, pos1, w1, h1 = calculate_bounding_box(state=[*tracking_agent_position,
                                                       *fleeing_agent_position,
                                                       *tracking_agent_orientation])
        self.assertEqual(round(result, 3), 0.837)

        # frame = np.zeros((1000, 1000))
        # frame[-pos0[1] - h0 // 2:-pos0[1] + h0 // 2,
        #       pos0[0] - w0 // 2:pos0[0] + h0 // 2] = 0.5
        # frame[-pos1[1] - h1 // 2:-pos1[1] + h1 // 2,
        #       pos1[0] - w1 // 2:pos1[0] + w1 // 2] = 1
        # plt.imshow(frame)
        # plt.show()
        # a = 100

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
