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
        "info": [],
        "observation": "forward_camera",
        "max_update_wait_period_s": 15,
        "store_action": True,
        "store_reward": True,
        "visible_xterm": True,
        "step_rate_fps": 15,
        "ros_launch_config": {
          "random_seed": 123,
          "robot_name": "turtlebot_sim",
          "fsm_config": "single_run",  # file with fsm params loaded from config/fsm
          "fsm": True,
          "control_mapping": True,
          "waypoint_indicator": True,
          "control_mapping_config": "default",
          "world_name": "debug",
          "x_pos": 0.0,
          "y_pos": 0.0,
          "z_pos": 0.0,
          "yaw_or": 1.57,
          "gazebo": True,
        },
        "actor_configs": [{
              "name": "ros_expert",
              "file": "src/sim/ros/config/actor/ros_expert.yml"
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

    def test_many_resets_and_wait_for_error(self):
        for count in range(100):
            print(f'episode: {count}')
            experience, observation = self._environment.reset()
            while experience.done == TerminationType.NotDone:
                experience, observation = self._environment.step()
                if experience.done == TerminationType.NotDone:
                    self.assertEqual(rospy.get_param('/world/reward/step'), experience.reward)
                else:
                    self.assertEqual(rospy.get_param('/world/reward/distance'), experience.reward)

    def tearDown(self) -> None:
        self._environment.remove()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
