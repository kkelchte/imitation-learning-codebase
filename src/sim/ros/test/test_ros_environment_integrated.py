import os
import shutil
import unittest

from datetime import datetime

import yaml

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
        # Step 1: check EnvironmentConfig
        self._environment = RosEnvironment(
            config=config
        )

    def test_waypoint_in_object_world(self):
        self.start_test('test_ros_environment.yml')
        # Step 2: start gzclient to see
        state = self._environment.reset()
        while state.terminal != TerminalType.Success and state.terminal != TerminalType.Failure:
            state = self._environment.step()
        self.assertEqual(state.terminal, TerminalType.Success)

    def tearDown(self) -> None:
        if hasattr(self, '_environment'):
            self._environment.remove()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
