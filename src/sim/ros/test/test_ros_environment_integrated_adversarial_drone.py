import os
import shutil
import time
import unittest

import numpy as np
import matplotlib.pyplot as plt
import rospy

from src.core.utils import get_filename_without_extension, get_to_root_dir
from src.core.data_types import TerminationType, Action, SensorType
from src.sim.common.environment import EnvironmentConfig
from src.sim.ros.src.ros_environment import RosEnvironment

MAXSTEPS = 10

config_dict = {
    "output_path": "/tmp",
    "factory_key": "ROS",
    "max_number_of_steps": MAXSTEPS,
    "ros_config": {
        "info": ['/tracking/cmd_vel', '/fleeing/cmd_vel', 'frame'],
        "observation": 'modified_state',
        "action_topic": 'python',
        "num_action_publishers": 2,
        "max_update_wait_period_s": 10,
        "visible_xterm": False,
        "step_rate_fps": 100,
        "ros_launch_config": {
          "random_seed": 123,
          "robot_name": "double_drone_sim",
          "fsm_mode": "TakeOverRun",  # file with fsm params loaded from config/fsm
          "fsm": True,
          "robot_display": False,
          "control_mapping": True,
          "control_mapping_config": "python_adversarial",
          "modified_state_publisher": True,
          "modified_state_publisher_mode": 'CombinedGlobalPoses',
          "modified_state_frame_visualizer": True,
          "waypoint_indicator": False,
          "world_name": "empty",
          "distance_tracking_fleeing_m": 10,
          "gazebo": True,
        },
        "actor_configs": [{
              "name": "altitude_control"
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

    def test_multiple_resets(self):
        time.sleep(rospy.get_param('/world/delay_evaluation') + 2)
        for _ in range(2):
            experience, observation = self._environment.reset()  # includes take off of two agents by fsm
            self.assertTrue(experience.action is None)
            self.assertEqual(experience.done, TerminationType.NotDone)
            count = 0
            while experience.done == TerminationType.NotDone:
                fake_adversarial_action = Action(actor_name='tracking_fleeing_agent',
                                                 value=np.asarray([0, 0, 5,  # tracking velocity
                                                                   0, 3, 0,  # fleeing velocity
                                                                   0,        # tracking angular z
                                                                   0]))      # fleeing angular z
                print(f'{count} {fake_adversarial_action}')
                experience, observation = self._environment.step(action=fake_adversarial_action)
                count += 1
                self.assertTrue(experience.observation is not None)
                self.assertTrue(experience.action is not None)
                self.assertTrue('frame' in experience.info.keys())
                self.assertEqual(experience.info['/tracking/cmd_vel'].value[2], 5)
                self.assertEqual(experience.info['/fleeing/cmd_vel'].value[1], 3)
            print(f'{count} vs {MAXSTEPS}')
            self.assertEqual(count, MAXSTEPS)
            # self.assertTrue('frame' in experience.info.keys())
            # plt.imshow(experience.info['frame'])
            # plt.show()

    def tearDown(self) -> None:
        self._environment.remove()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
