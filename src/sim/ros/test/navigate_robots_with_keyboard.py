import os
import shutil
import unittest

from src.core.utils import get_filename_without_extension
from src.core.data_types import TerminationType
from src.sim.common.environment import EnvironmentConfig
from src.sim.ros.src.ros_environment import RosEnvironment
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState


class TestRobots(unittest.TestCase):

    def start(self, robot_name: str, fsm_config: str = 'single_run') -> None:
        self.output_dir = f'test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        config_dict = {
            'output_path': self.output_dir,
            'factory_key': "ROS",
            'max_number_of_steps': -1,
            'ros_config': {
                'ros_launch_config': {
                    'control_mapping_config': 'keyboard',
                    'fsm_config': fsm_config,
                    'gazebo': 'sim' in robot_name,
                    'random_seed': 123,
                    'robot_name': robot_name,
                    'world_name': 'debug_turtle' if 'sim' in robot_name else 'empty',
                    'robot_display': True,
                    'x_pos': 0.0,
                    'y_pos': 0.0,
                    'yaw_or': 1.57,
                    'z_pos': 0.0,
                },
                'actor_configs': [
                    {
                        'file': f'src/sim/ros/config/actor/keyboard_{robot_name}.yml',
                        'name': 'keyboard'
                    }
                ],
                'visible_xterm': True,
            },
        }
        config = EnvironmentConfig().create(config_dict=config_dict)
        self._environment = RosEnvironment(config=config)

    def test_turtlebot_sim(self):
        self.start(robot_name='turtlebot_sim', fsm_config='takeover_run')
        experience, _ = self._environment.reset()
        # wait delay evaluation time
        while experience.done == TerminationType.Unknown:
            experience = self._environment.step()

        while self._environment.fsm_state != FsmState.Terminated:
            _ = self._environment.step()

    def test_drone_sim(self):
        self.start(robot_name='drone_sim', fsm_config='takeover_run')
        self._environment.reset()
        while True:
            self._environment.step()

    def test_bebop_real(self):
        self.start(robot_name='bebop_real', fsm_config='takeover_run')
        self._environment.reset()
        while True:
            self._environment.step()

    def test_tello_real(self):
        self.start(robot_name='tello_real', fsm_config='takeover_run')
        self._environment.reset()
        while True:
            self._environment.step()

    def tearDown(self) -> None:
        if hasattr(self, '_environment'):
            self._environment.remove()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
