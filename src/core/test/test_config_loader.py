import unittest
import os
import shutil

import yaml
from datetime import datetime
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
    store_data: bool = None


class TestConfigLoader(unittest.TestCase):

    def setUp(self) -> None:
        self.TEST_DIR = f'test-{datetime.strftime(datetime.now(), "%d-%m-%y_%H-%M")}'
        if not os.path.exists(self.TEST_DIR):
            os.makedirs(self.TEST_DIR)
        self.config_file = 'src/core/test/config/test_config_loader_config.yml'
        if not os.path.exists(self.config_file):
            raise FileNotFoundError('Are you in the code root directory?')
        with open(self.config_file, 'r') as f:
            self.config_dict = yaml.load(f, Loader=yaml.FullLoader)
        self.config_dict['output_path'] = self.TEST_DIR

    def test_load_from_dict(self):
        config = DummyDataCollectionConfig().create(config_dict=self.config_dict)

        print(config)
        self.assertEqual(config.store_data, True)
        self.assertEqual(config.environment_config.number_of_runs, 10)
        self.assertTrue(isinstance(config.model_config, DummyModelConfig))

    def test_load_from_file(self):
        config = DummyDataCollectionConfig().create(config_file=self.config_file)

        print(config)
        self.assertEqual(config.store_data, True)
        self.assertEqual(config.environment_config.number_of_runs, 10)
        self.assertTrue(isinstance(config.model_config, DummyModelConfig))

    def test_error_for_none_value(self):
        del self.config_dict['store_data']
        with self.assertRaises(Exception):
            DummyDataCollectionConfig().create(config_dict=self.config_dict)

        del self.config_dict['environment_config']['environment_name']
        with self.assertRaises(Exception):
            DummyDataCollectionConfig().create(config_dict=self.config_dict)

    def test_usage_of_default_value(self):
        del self.config_dict['environment_config']['number_of_runs']

        config = DummyDataCollectionConfig().create(config_dict=self.config_dict)
        print(config)
        self.assertEqual(config.environment_config.number_of_runs, 5)

    def test_raise_unknown_value(self):
        self.config_dict['new_key'] = 'new_value'
        with self.assertRaises(Exception):
            DummyDataCollectionConfig().create(config_dict=self.config_dict)

    def test_config_parser(self):
        config_file = 'src/core/test/config/test_config_loader_config.yml'
        arguments = Parser().parse_args(["--config", config_file])
        config = DummyDataCollectionConfig().create(config_file=arguments.config)
        print(config)

    def test_config_output_path(self):
        config = DummyDataCollectionConfig().create(config_dict=self.config_dict)
        self.assertEqual(config.output_path, config.model_config.output_path)
        self.assertEqual(config.output_path, config.environment_config.output_path)

    def test_saved_config_in_output_path(self):
        config = DummyDataCollectionConfig().create(config_dict=self.config_dict)
        new_config_file = os.path.join(self.TEST_DIR, 'configs', 'dummy_data_collection_config.yml')
        self.assertTrue(os.path.isfile(new_config_file))
        restored_config = DummyDataCollectionConfig().create(config_file=new_config_file)
        self.assertEqual(restored_config, config)

    def tearDown(self):
        shutil.rmtree(self.TEST_DIR, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
