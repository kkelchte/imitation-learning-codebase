import shutil
import unittest
import os

from datetime import datetime
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from dataclasses_json.undefined import UndefinedParameterError

from src.core.object_factory import ObjectFactory, ConfigFactory
from src.core.config_loader import Config
from src.core.utils import get_filename_without_extension


@dataclass_json
@dataclass
class DummyEnvironmentConfig(Config):
    output_path: str = ''
    factory_key: str = None
    episode_len: int = None


@dataclass_json
@dataclass
class DummyGymEnvironmentConfig(DummyEnvironmentConfig):
    reward_type: str = None


@dataclass_json
@dataclass
class DummyGazeboEnvironmentConfig(DummyEnvironmentConfig):
    robot_type: str = None


class DummyEnvironmentConfigFactory(ConfigFactory):

    def __init__(self):
        self._class_dict = {
            'gym': DummyGymEnvironmentConfig,
            'gazebo': DummyGazeboEnvironmentConfig
        }
        super().__init__(self._class_dict)


class DummyGymEnvironment:

    def __init__(self, config: DummyGymEnvironmentConfig):
        if not config.reward_type:
            raise IOError('Configuration not successful.')


class DummyGazeboEnvironment:

    def __init__(self, config: DummyGazeboEnvironmentConfig):
        if not config.robot_type:
            raise IOError('Configuration not successful.')


class DummyEnvironmentFactory(ObjectFactory):

    def __init__(self):
        self._class_dict = {
            'gym': DummyGymEnvironment,
            'gazebo': DummyGazeboEnvironment
        }
        super().__init__(self._class_dict)


class TestObjectFactory(unittest.TestCase):

    def setUp(self) -> None:
        self.TEST_DIR = f'test_dir/{get_filename_without_extension(__file__)}'
        if not os.path.exists(self.TEST_DIR):
            os.makedirs(self.TEST_DIR)
        self.config = {
            'output_path': self.TEST_DIR,
            'episode_len': 10
        }

    def test_gym_config_case(self):
        self.config['factory_key'] = 'gym'
        self.config['reward_type'] = 'dense'
        config = DummyEnvironmentConfigFactory().create(config=self.config)
        self.assertTrue(isinstance(config, DummyGymEnvironmentConfig))

    def test_gym_failure_config_case(self):
        self.config['factory_key'] = 'gazebo'
        self.config['reward_type'] = 'dense'

        with self.assertRaises(UndefinedParameterError):
            DummyEnvironmentConfigFactory().create(config=self.config)

    def test_create_gazebo_instant(self):
        self.config['factory_key'] = 'gazebo'
        self.config['robot_type'] = 'turtle'

        config = DummyEnvironmentConfigFactory().create(config=self.config)
        instant = DummyEnvironmentFactory().create(config=config)
        self.assertTrue(instant, DummyGazeboEnvironment)

    def tearDown(self):
        shutil.rmtree(self.TEST_DIR, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
