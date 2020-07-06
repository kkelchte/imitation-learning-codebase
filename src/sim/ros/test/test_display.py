import os
import shutil
import unittest
import time
from typing import List

import rospy
from dataclasses import dataclass

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty, Float32, String  # Do not remove!
from sensor_msgs.msg import LaserScan  # Do not remove!

from imitation_learning_ros_package.msg import RosReward

from src.core.utils import get_filename_without_extension
from src.sim.ros.catkin_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.core.data_types import TerminationType
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.test.common_utils import TopicConfig, TestPublisherSubscriber

""" Test FSM in take off mode with reward for multiple resets
"""


class TestDisplay(unittest.TestCase):

    def setUp(self) -> None:
        self.config = {
            'robot_name': 'drone_sim',
            'world_name': 'debug_drone',
            'gazebo': True,
            'robot_display': True,
            'fsm': False,
            'robot_mapping': False,
            'waypoint_indicator': False,
        }
        self.output_dir = f'{os.environ["DATADIR"] if "DATADIR" in os.environ.keys() else os.environ["HOME"]}' \
                          f'/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        self.config['output_path'] = self.output_dir

        # spinoff roslaunch
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=self.config,
                                       visible=True)

    def test_camera_feed(self):
        start_time = time.time()
        max_duration = 60
        while (time.time() - start_time) < max_duration:
            time.sleep(1)

    def tearDown(self) -> None:
        self._ros_process.terminate()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
