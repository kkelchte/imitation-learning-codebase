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

WORLDNAME = 'line_worlds/model_009'
config_dict = {
    "output_path": "/tmp",
    "factory_key": "ROS",
    "max_number_of_steps": -1,
    "ros_config": {
        "info": [
            "sensor/odometry"
        ],
        "observation": "forward_camera",
        "max_update_wait_period_s": 120,
        "store_action": True,
        "store_reward": False,
        "visible_xterm": True,
        "step_rate_fps": 30,
        "ros_launch_config": {
          "random_seed": 123,
          "robot_name": "drone_sim",
          "fsm_config": "single_run",  # file with fsm params loaded from config/fsm
          "fsm": True,
          "control_mapping": True,
          "waypoint_indicator": True,
          "control_mapping_config": "noisy_line_worlds",  # default
          "world_name": WORLDNAME,
          "x_pos": 0.0,
          "y_pos": 0.0,
          "z_pos": 0.5,
          "yaw_or": 1.57,
          "gazebo": True,
        },
        "actor_configs": [{
              "name": "ros_expert",
              "file": "src/sim/ros/config/actor/ros_expert_wp_slow.yml"
            }],
    }
}


class ValidateExpert(unittest.TestCase):

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

    def test_multiple_runs(self):
        for _ in range(1):
            experience, observation = self._environment.reset()
            self.assertTrue(experience.action is None)
            self.assertEqual(experience.done, TerminationType.NotDone)
            count = 0
            while experience.done == TerminationType.NotDone:
                count += 1
                experience, observation = self._environment.step()
            print(f'finished with {experience.done.name}')

    def tearDown(self) -> None:
        self._environment.remove()
        #shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
