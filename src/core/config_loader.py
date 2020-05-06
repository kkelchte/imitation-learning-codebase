import os
import argparse
from enum import IntEnum
from typing import Optional

import yaml
from dataclasses import dataclass
from dataclasses_json import dataclass_json, Undefined

from src.core.utils import camelcase_to_snake_format, get_date_time_tag


def iterative_add_output_path(dictionary: dict, output_path: str) -> dict:
    if 'output_path' not in dictionary.keys():
        dictionary['output_path'] = output_path
    for key, value in dictionary.items():
        if isinstance(value, Config):
            dictionary[key] = iterative_add_output_path(value.__dict__, output_path)
    return dictionary


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class Config:
    output_path: Optional[str] = None
    """Define config class to translate yaml/dicts to corresponding config objects.

    Based on dataclass_json object.
    Raises error if field contains None.
    """

    def create(self,
               config_dict: dict = None,
               config_file: str = '',
               store: bool = True):
        assert not (config_file != '' and config_dict is not None)
        assert (config_file != '' or config_dict is not None)

        if config_file:
            if not os.path.exists(config_file):
                raise FileNotFoundError('Are you in the code root directory?')
            with open(config_file, 'r') as f:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
        instant = self.from_dict(config_dict)
        if not instant.output_path.startswith('/'):
            instant.output_path = f'{os.environ["HOME"]}/{instant.output_path}'
        instant.iterative_add_output_path(output_path=instant.output_path)
        instant.post_init()
        instant.iterative_check_for_none()
        if store:
            instant.save_config_file()
        return instant

    def iterative_check_for_none(self) -> None:
        for key, value in self.__dict__.items():
            if value is None:
                raise IOError(f'Found None value in config. \n {key}: {value}')
            if isinstance(value, Config):
                value.iterative_check_for_none()
            if isinstance(value, list):
                for element in value:
                    if isinstance(element, Config):
                        element.iterative_check_for_none()

    def iterative_add_output_path(self, output_path: str) -> None:
        if self.output_path is None:
            self.output_path = output_path
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                value.iterative_add_output_path(self.output_path)
            if isinstance(value, list):
                for element in value:
                    if isinstance(element, Config):
                        element.iterative_add_output_path(self.output_path)

    def save_config_file(self) -> str:
        os.makedirs(os.path.join(self.output_path, 'configs'), exist_ok=True)
        config_file_path = os.path.join(self.output_path, 'configs',
                                        f'{get_date_time_tag()}_'
                                        f'{camelcase_to_snake_format(self.__class__.__name__)}.yml')
        with open(config_file_path, 'w') as f:
            yaml.dump(data=self.yaml_approved_dict(),
                      stream=f)
        return config_file_path

    def yaml_approved_dict(self) -> dict:
        output_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                output_dict[key] = value.yaml_approved_dict()
            else:
                if isinstance(value, IntEnum):
                    output_dict[key] = int(value)
                else:
                    output_dict[key] = value
        return output_dict

    def post_init(self):
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                value.post_init()
            if isinstance(value, list):
                for element in value:
                    if isinstance(element, Config):
                        element.post_init()


class Parser(argparse.ArgumentParser):
    """Parser class to get config retrieve config file argument"""

    def __init__(self):
        super().__init__()
        self.add_argument("--config", type=str, default=None)
