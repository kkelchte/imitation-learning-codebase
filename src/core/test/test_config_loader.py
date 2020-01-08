import unittest
import os

import yaml
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from src.core.config_loader import Config, Parser


@dataclass_json
@dataclass
class DummyModelConfig(Config):
    model_checkpoint: str = None
    model_path: str = None



@dataclass_json
@dataclass
class DummyEnvironmentConfig(Config):
    environment_name: str = None
    number_of_runs: int = 5


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

    def test_error_for_none_value(self):
        config_file = 'src/core/test/config/test_config_loader_config.yml'
        if not os.path.exists(config_file):
            raise FileNotFoundError('Are you in the code root directory?')
        with open(config_file, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)

        del config_dict['output_path']
        with self.assertRaises(Exception):
            DummyDataCollectionConfig().create(config_dict=config_dict)

    def test_usage_of_default_value(self):
        config_file = 'src/core/test/config/test_config_loader_config.yml'
        if not os.path.exists(config_file):
            raise FileNotFoundError('Are you in the code root directory?')
        with open(config_file, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)

        del config_dict['environment_config']['number_of_runs']

        config = DummyDataCollectionConfig().create(config_dict=config_dict)
        print(config)
        self.assertEqual(config.environment_config.number_of_runs, 5)

    def test_config_parser(self):
        config_file = 'src/core/test/config/test_config_loader_config.yml'
        arguments = Parser().parse_args(["--config", config_file])
        config = DummyDataCollectionConfig().create(config_file=arguments.config)


if __name__ == '__main__':
    unittest.main()
