import os
import shutil
import unittest

import rospy
import yaml

from src.core.utils import get_filename_without_extension
from src.sim.common.data_types import TerminationType
from src.sim.common.environment import EnvironmentConfig
from src.sim.ros.src.ros_environment import RosEnvironment
from src.sim.ros.catkin_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState

########################
# Settings
########################
# world_name = 'cube_world'
# robot_name = 'turtlebot_sim'
# fsm_config = 'single_run'

world_name = 'object_world'
robot_name = 'drone_sim'
fsm_config = 'takeoff_run'


class TestRosExpert(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'test_dir/{get_filename_without_extension(__file__)}'
        config_file = 'test_expert_in_environment'
        os.makedirs(self.output_dir, exist_ok=True)
        with open(f'src/sim/ros/test/config/{config_file}.yml', 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        config_dict['output_path'] = self.output_dir
        config_dict['ros_config']['ros_launch_config']['world_name'] = world_name
        config_dict['ros_config']['ros_launch_config']['robot_name'] = robot_name
        config_dict['ros_config']['ros_launch_config']['fsm_config'] = fsm_config
        self.environment_config = EnvironmentConfig().create(
            config_dict=config_dict
        )

    def test_expert(self):
        self.environment = RosEnvironment(
            config=self.environment_config
        )
        experience = self.environment.reset()

        # wait delay evaluation time
        while experience.done == TerminationType.Unknown:
            experience = self.environment.step()
        print(f'finished startup')
        waypoints = rospy.get_param('/world/waypoints')

        for waypoint_index, waypoint in enumerate(waypoints[:-1]):
            print(f'started with waypoint: {waypoint}')
            while experience.info['current_waypoint'].tolist() == waypoint:
                experience = self.environment.step()

        print(f'ending with waypoint {waypoints[-1]}')
        while not self.environment.fsm_state == FsmState.Terminated:
            experience = self.environment.step()
        print(f'terminal type: {experience.done.name}')

    def tearDown(self) -> None:
        self.environment.remove()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
