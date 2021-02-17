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
        "max_update_wait_period_s": 10,
        "store_action": True,
        "store_reward": True,
        "visible_xterm": False,
        "step_rate_fps": 100,
        "ros_launch_config": {
          "random_seed": 123,
          "robot_name": "drone_sim",
          "fsm_config": "single_run",  # file with fsm params loaded from config/fsm
          "fsm": True,
          "robot_display": False,
          "control_mapping": True,
          "waypoint_indicator": True,
          "control_mapping_config": "debug",
          "world_name": "debug_drone",
          "x_pos": 0.0,
          "y_pos": 0.0,
          "z_pos": 1.5,
          "yaw_or": 1.57,
          "gazebo": True,
        },
        "actor_configs": [{
              "name": "ros_expert",
              "file": "src/sim/ros/config/actor/ros_expert_wp.yml"
            }],
    }
}


class TestRosIntegrated(unittest.TestCase):

    def setUp(self) -> None:
        config_dict['output_path'] = f'test_dir/{get_filename_without_extension(__file__)}'
        config = EnvironmentConfig().create(
            config_dict=config_dict
        )
        self.output_dir = config.output_path
        self._environment = RosEnvironment(
            config=config
        )
        time.sleep(1)

    def test_multiple_resets(self):
        time.sleep(rospy.get_param('/world/delay_evaluation') + 2)
        for _ in range(2):
            experience, observation = self._environment.reset()
            self.assertTrue(experience.action is None)
            self.assertEqual(experience.done, TerminationType.NotDone)
            count = 0
            while experience.done == TerminationType.NotDone:
                experience, observation = self._environment.step()
                count += 1
                self.assertTrue(experience.observation is not None)
                self.assertTrue(experience.action is not None)
                if experience.done == TerminationType.NotDone:
                    self.assertEqual(experience.reward, rospy.get_param('/world/reward/step'))
                else:
                    self.assertEqual(experience.reward, rospy.get_param('/world/reward/goal'))
            self.assertGreater(np.sum(experience.info['odometry'][:3]), 1)
            self.assertEqual(experience.done, TerminationType.Success)

    def tearDown(self) -> None:
        self._environment.remove()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
