import unittest

from src.scripts.run_experiment import ExperimentConfig
from src.sim.common.environment_runner import EnvironmentRunner


class TestRosExperiments(unittest.TestCase):

    def test_ros_without_data_collection(self):
        config_file = 'src/scripts/test/config/test_data_collection_in_ros_config.yml'
        config = ExperimentConfig().create(config_file=config_file)
        environment_runner = EnvironmentRunner(config=config.runner_config)
        environment_runner.run()
        # self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
