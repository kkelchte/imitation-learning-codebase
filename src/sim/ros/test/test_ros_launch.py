#!/usr/bin/python3.7
import os
import shutil
import unittest
import time

import subprocess
import rospy
import numpy as np

from src.core.utils import count_grep_name
from src.core.utils import get_filename_without_extension
from src.sim.common.actors import ActorConfig
from src.sim.common.data_types import TerminationType, EnvironmentType, Action, ProcessState
from src.sim.common.environment import EnvironmentConfig, RosConfig, RosLaunchConfig
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.src.ros_environment import RosEnvironment


class TestRos(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'test_dir/{get_filename_without_extension(__file__)}'
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

    #@unittest.skip
    def test_launch_and_terminate_gazebo(self):
        random_seed = 123
        config = {
            'random_seed': random_seed,
            'gazebo': 'true',
            'world_name': 'empty_world',
            'output_path': self.output_dir
        }
        ros_process = RosWrapper(launch_file='load_ros.launch',
                                 config=config,
                                 visible=False)
        self.assertEqual(ros_process.get_state(), ProcessState.Running)
        time.sleep(5)
        self.assertTrue(count_grep_name('gzserver') >= 1)
        ros_process.terminate()
        self.assertTrue(count_grep_name('gzserver') == 0)

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

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
