import os
import shutil
import time
import unittest

import numpy as np
import rospy

from src.core.utils import get_filename_without_extension, get_to_root_dir
from src.core.data_types import TerminationType
from src.sim.common.environment import EnvironmentConfig
from src.sim.ros.src.ros_environment import RosEnvironment


config_dict = {
    "output_path": "/tmp",
    "factory_key": "ROS",
    "max_number_of_steps": -1,
    "ros_config": {
        "info": [
            "current_waypoint",
            "sensor/odometry"
        ],
        "observation": "forward_camera",
        "visible_xterm": False,
        "step_rate_fps": 10,
        "ros_launch_config": {
          "random_seed": 123,
          "robot_name": "turtlebot_sim",
          "fsm_config": "single_run",  # file with fsm params loaded from config/fsm
          "fsm": True,
          "control_mapping": True,
          "waypoint_indicator": True,
          "control_mapping_config": "debug",
          "world_name": "object_world",
          "x_pos": 0.0,
          "y_pos": 0.0,
          "z_pos": 0.0,
          "yaw_or": 1.57,
          "gazebo": True,
        },
        "actor_configs": [{
              "name": "ros_expert",
              "file": "src/sim/ros/config/actor/ros_expert_noisy.yml"
            }],
    }
}


class TestRosIntegrated(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        config_dict['output_path'] = self.output_dir
        config = EnvironmentConfig().create(
            config_dict=config_dict
        )
        self._environment = RosEnvironment(
            config=config
        )

    def test_multiple_resets(self):
        waypoints = rospy.get_param('/world/waypoints')
        for _ in range(2):
            experience = self._environment.reset()
            # wait delay evaluation time
            while experience.done == TerminationType.Unknown:
                experience = self._environment.step()
            self.assertEqual(waypoints[0], experience.info['current_waypoint'].tolist())
            self.assertLess(np.sum(experience.info['odometry'][:3]), 0.2)
            count = 0
            while experience.done == TerminationType.NotDone:
                count += 1
                experience = self._environment.step()
                self.assertNotEqual(experience.observation, self._environment._default_observation)
                self.assertNotEqual(len(experience.action.value),
                                    len(self._environment._default_action.value))
            self.assertGreater(np.sum(experience.info['odometry'][:3]), 0.5)
            self.assertEqual(experience.done, TerminationType.Success)

    def tearDown(self) -> None:
        self._environment.remove()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
