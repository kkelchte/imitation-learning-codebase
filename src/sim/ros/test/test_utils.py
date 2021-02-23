import os
import shutil
import unittest

import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import PointStamped, PoseStamped, Pose, Point, Quaternion
from scipy.spatial.transform import Rotation as R

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

    def test_transform_points(self):
        pos_a = PointStamped()
        pos_a.header.frame_id = 'world_a'
        pos_a.point.x = 1
        pos_b = PointStamped()
        pos_b.header.frame_id = 'world_b'
        pos_b.point.y = 1
        pos_a.point, pos_b.point = transform([pos_a.point, pos_b.point],
                                             R.from_euler('XYZ', (0, 0, 3.14/2),
                                                          degrees=False).as_matrix())
        results_array = transform([np.asarray([1, 0, 0]), np.asarray([0, 1, 0])],
                                  R.from_euler('XYZ', (0, 0, 3.14/2),
                                               degrees=False).as_matrix())

        self.assertAlmostEqual(pos_a.point.x, results_array[0][0])
        self.assertAlmostEqual(pos_a.point.y, results_array[0][1])
        self.assertAlmostEqual(pos_a.point.z, results_array[0][2])

        self.assertAlmostEqual(pos_b.point.x, results_array[1][0])
        self.assertAlmostEqual(pos_b.point.y, results_array[1][1])
        self.assertAlmostEqual(pos_b.point.z, results_array[1][2])

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
