import os
import shutil
import unittest

from src.core.utils import get_filename_without_extension
from src.core.data_types import ProcessState, TerminationType
from src.sim.common.environment import EnvironmentConfig
from src.sim.gym.gym_environment import GymEnvironment


class TestGymEnvironment(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

    def _test_environment_by_name(self, name: str) -> bool:
        config = {
            'output_path': self.output_dir,
            'factory_key': "GYM",
            'max_number_of_steps': 200,
            'ros_config': None,
            'gym_config': {
                'random_seed': 123,
                'world_name': name,
                'render': False
            }
        }

        environment = GymEnvironment(EnvironmentConfig().create(config_dict=config))
        experience, _ = environment.reset()
        while experience.done != TerminationType.Done:
            experience, _ = environment.step(environment.get_random_action())
        return environment.remove() == ProcessState.Terminated

    def test_environments(self):
        self.assertTrue(True, self._test_environment_by_name('CartPole-v0'))
        self.assertTrue(True, self._test_environment_by_name('Pendulum-v0'))
        self.assertTrue(True, self._test_environment_by_name('MountainCarContinuous-v0'))
        self.assertTrue(True, self._test_environment_by_name('Pong-v0'))

    def tearDown(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
