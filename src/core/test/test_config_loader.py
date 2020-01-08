import unittest
import os

import yaml
import json
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from src.core.config_loader import Config


@dataclass_json
@dataclass
class DummyModelConfig(Config):
    model_path: str
    model_checkpoint: str = None


@dataclass_json
@dataclass
class DummyEnvironmentConfig(Config):
    environment_name: str
    number_of_runs: int


@dataclass_json
@dataclass
class DummyDataCollectionConfig(Config):
    output_path: str = None
    model_config: DummyModelConfig = None
    environment_config: DummyEnvironmentConfig = None
    store_data: bool = True


class TestConfigLoader(unittest.TestCase):

    def test_load_from_dict(self):
        config_file = 'src/core/test/config/test_config_loader_config.yml'
        if not os.path.exists(config_file):
            raise FileNotFoundError('Are you in the code root directory?')
        with open(config_file, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)

        config = DummyDataCollectionConfig().create(config_dict=config_dict)

        print(config)

        self.assertEqual(config.store_data, True)
        self.assertEqual(config.environment_config.number_of_runs, 10)
        self.assertTrue(isinstance(config.model_config, DummyModelConfig))

    def test_load_from_file(self):
        config_file = 'src/core/test/config/test_config_loader_config.yml'
        config = DummyDataCollectionConfig().create(config_file=config_file)

        print(config)

        self.assertEqual(config.store_data, True)
        self.assertEqual(config.environment_config.number_of_runs, 10)
        self.assertTrue(isinstance(config.model_config, DummyModelConfig))


if __name__ == '__main__':
    unittest.main()
