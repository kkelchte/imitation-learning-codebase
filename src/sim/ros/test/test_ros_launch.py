#!/usr/bin/python3.8
import os
import shutil
import unittest
import time

import rospy

from src.core.utils import count_grep_name
from src.core.utils import get_filename_without_extension
from src.core.data_types import ProcessState
from src.sim.ros.src.process_wrappers import RosWrapper


class TestRos(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["DATADIR"] if "DATADIR" in os.environ.keys() else os.environ["HOME"]}' \
                          f'/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

    @unittest.skip
    def test_launch_and_terminate_ros(self):
        ros_process = RosWrapper(launch_file='empty_ros.launch',
                                 config={
                                     'output_path': self.output_dir
                                 })
        self.assertEqual(ros_process.get_state(), ProcessState.Running)
        self.assertTrue(count_grep_name('ros') > 0)
        ros_process.terminate()

    @unittest.skip
    def test_launch_and_terminate_gazebo(self):
        random_seed = 123
        config = {
            'random_seed': random_seed,
            'gazebo': 'true',
            'world_name': 'cube_world',
            'output_path': self.output_dir
        }
        ros_process = RosWrapper(launch_file='load_ros.launch',
                                 config=config,
                                 visible=False)
        self.assertEqual(ros_process.get_state(), ProcessState.Running)
        time.sleep(5)
        self.assertGreaterEqual(count_grep_name('gzserver'), 1)
        ros_process.terminate()
        self.assertEqual(count_grep_name('gzserver'), 0)

    @unittest.skip
    def test_load_params(self):
        config = {
            'robot_name': 'turtlebot_sim',
            'fsm': False,
            'output_path': self.output_dir
        }
        ros_process = RosWrapper(launch_file='load_ros.launch',
                                 config=config,
                                 visible=True)
        self.assertEqual(rospy.get_param('/robot/forward_camera_topic'), '/wa/camera/image_raw')
        ros_process.terminate()

    def test_full_config(self):
        config = {
          "output_path": self.output_dir,
          "random_seed": 123,
          "robot_name": "turtlebot_sim",
          "fsm_config": "single_run",  # file with fsm params loaded from config/fsm
          "fsm": True,
          "control_mapping": True,
          "waypoint_indicator": True,
          "control_mapping_config": "debug",
          "world_name": "debug_turtle",
          "x_pos": 0.0,
          "y_pos": 0.0,
          "z_pos": 0.0,
          "yaw_or": 1.57,
          "gazebo": True,
        }
        ros_process = RosWrapper(launch_file='load_ros.launch',
                                 config=config,
                                 visible=False)
        ros_process.terminate()

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
