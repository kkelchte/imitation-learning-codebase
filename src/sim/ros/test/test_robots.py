import os
import shutil
import unittest

from src.core.utils import get_filename_without_extension
from src.sim.common.data_types import TerminalType
from src.sim.common.environment import EnvironmentConfig
from src.sim.ros.src.ros_environment import RosEnvironment
from src.sim.ros.catkin_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState


class TestRobots(unittest.TestCase):

    def start(self, robot_name: str, fsm_config: str = 'single_run') -> None:
        self.output_dir = f'test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        config_dict = {
            'output_path': self.output_dir,
            'factory_key': 0,
            'max_number_of_steps': -1,
            'ros_config': {
                'ros_launch_config': {
                    'control_mapping_config': 'keyboard',
                    'fsm_config': fsm_config,
                    'gazebo': True,
                    'random_seed': 123,
                    'robot_name': robot_name,
                    'world_name': 'object_world',
                    'x_pos': 0.0,
                    'y_pos': 0.0,
                    'yaw_or': 1.57,
                    'z_pos': 0.0,
                },
                'visible_xterm': True,
            },
            'actor_configs': [
                {
                    'file': f'src/sim/ros/config/keyboard/{robot_name}.yml',
                    'name': 'keyboard',
                    'type': 2
                }
            ],
        }
        config = EnvironmentConfig().create(config_dict=config_dict)
        self._environment = RosEnvironment(config=config)

    def test_turtlebot_sim(self):
        self.start(robot_name='turtlebot_sim')
        state = self._environment.reset()
        # wait delay evaluation time
        while state.terminal == TerminalType.Unknown:
            state = self._environment.step()

        while self._environment.fsm_state != FsmState.Terminated:
            _ = self._environment.step()

    def test_drone_sim(self):
        self.start(robot_name='drone_sim', fsm_config='takeoff_run')
        state = self._environment.reset()
        # wait delay evaluation time
        while state.terminal == TerminalType.Unknown:
            state = self._environment.step()

        while self._environment.fsm_state != FsmState.Terminated:
            _ = self._environment.step()

    def tearDown(self) -> None:
        if hasattr(self, '_environment'):
            self.assertTrue(self._environment.remove())
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
