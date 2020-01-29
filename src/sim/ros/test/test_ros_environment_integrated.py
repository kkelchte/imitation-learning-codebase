import os
import shutil
import unittest

from datetime import datetime

import rospy
import yaml

from src.sim.ros.catkin_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.common.data_types import TerminalType
from src.sim.common.environment import EnvironmentConfig
from src.sim.ros.src.ros_environment import RosEnvironment


class TestRosIntegrated(unittest.TestCase):

    def start_test(self, config_file: str) -> None:
        self.output_dir = f'tmp_test_dir/{datetime.strftime(datetime.now(), format="%y-%m-%d_%H-%M-%S")}'
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

    def test_waypoints_in_object_world(self):
        self.start_test('test_ros_environment')
        # Step 2: start gzclient to see

        state = self._environment.reset()
        # wait delay evaluation time
        while self._environment.fsm_state == FsmState.Unknown:
            state = self._environment.step()

        waypoints = rospy.get_param('/world/waypoints')
        self.assertEqual(waypoints[0], state.sensor_data['current_waypoint'].tolist())
        for waypoint_index, waypoint in enumerate(waypoints):
            while state.sensor_data['current_waypoint'].tolist() == waypoint:
                state = self._environment.step()
                self.assertTrue(state.terminal != TerminalType.Failure)
            # assert transition to next waypoint occurs
            self.assertEqual(state.sensor_data['current_waypoint'].tolist(),
                             waypoints[(waypoint_index + 1) % len(waypoints)])
        # all waypoints should be reached and environment should have reach success state
        self.assertEqual(state.terminal, TerminalType.Success)

    def tearDown(self) -> None:
        if hasattr(self, '_environment'):
            self._environment.remove()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
