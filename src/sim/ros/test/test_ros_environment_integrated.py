import os
import shutil
import unittest

import numpy as np
import rospy
import yaml

from src.core.utils import get_filename_without_extension
from src.core.data_types import TerminationType
from src.sim.common.environment import EnvironmentConfig
from src.sim.ros.src.ros_environment import RosEnvironment


class TestRosIntegrated(unittest.TestCase):

    def setUp(self) -> None:
        config_file = 'test_ros_environment'
        self.output_dir = f'test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        with open(f'src/sim/ros/test/config/{config_file}.yml', 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        config_dict['output_path'] = self.output_dir
        with open(os.path.join(self.output_dir, 'config.yml'), 'w') as f:
            yaml.dump(config_dict, f)
        config = EnvironmentConfig().create(
            config_file=os.path.join(self.output_dir, 'config.yml')
        )
        self._environment = RosEnvironment(
            config=config
        )

    # def test_waypoints_in_object_world(self):
    #     experience = self._environment.reset()
    #     # wait delay evaluation time
    #     while experience.done == TerminationType.Unknown:
    #         experience = self._environment.step()
    #     waypoints = rospy.get_param('/world/waypoints')
    #     self.assertEqual(waypoints[0], experience.info['current_waypoint'].tolist())
    #     for waypoint_index, waypoint in enumerate(waypoints[:-1]):
    #         while experience.info['current_waypoint'].tolist() == waypoint:
    #             experience = self._environment.step()
    #             self.assertTrue(experience.done != TerminationType.Failure)
    #         # assert transition to next waypoint occurs
    #         self.assertEqual(experience.info['current_waypoint'].tolist(),
    #                          waypoints[(waypoint_index + 1) % len(waypoints)])
    #     while not self._environment.fsm_state == FsmState.Terminated:
    #         experience = self._environment.step()
    #     # all waypoints should be reached and environment should have reach success experience
    #     self.assertEqual(experience.done, TerminationType.Success)

    #@unittest.skip
    def test_multiple_resets(self):
        waypoints = rospy.get_param('/world/waypoints')
        for _ in range(2):
            experience = self._environment.reset()
            # wait delay evaluation time
            while experience.done == TerminationType.Unknown:
                experience = self._environment.step()
            self.assertEqual(waypoints[0], experience.info['current_waypoint'].tolist())
            self.assertTrue(np.sum(experience.info['odometry'][:3]) < 0.2)
            while experience.done == TerminationType.NotDone:
                experience = self._environment.step()
            self.assertTrue(np.sum(experience.info['odometry'][:3]) > 0.5)
            self.assertEqual(experience.done, TerminationType.Success)

    def tearDown(self) -> None:
        self._environment.remove()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
