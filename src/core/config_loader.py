import os
import argparse

import yaml
from dataclasses import dataclass
from dataclasses_json import dataclass_json


def iterative_add_output_path(dictionary: dict, output_path: str) -> dict:
    if 'output_path' not in dictionary.keys():
        dictionary['output_path'] = output_path
    for key, value in dictionary.items():
        if isinstance(value, Config):
            dictionary[key] = iterative_add_output_path(value.__dict__, output_path)
    return dictionary


@dataclass_json
@dataclass
class Config:
    output_path: str = None
    """Define config class to translate yaml/dicts to corresponding config objects.

    Based on dataclass_json object.
    Raises error if field contains None.
    """

    def create(self,
               config_dict: dict = {},
               config_file: str = ''):
        assert not (config_file and config_dict)
        assert (config_dict or config_file)

        if config_file:
            if not os.path.exists(config_file):
                raise FileNotFoundError('Are you in the code root directory?')
            with open(config_file, 'r') as f:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)

        instant = self.from_dict(config_dict)
        instant.iterative_add_output_path(output_path=instant.output_path)
        instant.iterative_check_for_none()
        # instant = iterative_add_output_path(instant, output_path=instant.output_path)
        # iterative_check_for_none_values(instant)
        # instant = self.from_dict(
        #     iterative_add_output_path(instant.__dict__, output_path=instant.output_path)
        # )
        return instant

    def iterative_check_for_none(self) -> None:
        for key, value in self.__dict__.items():
            if value is None:
                raise IOError(f'Found None value in config. \n {key}: {value}')
            if isinstance(value, Config):
                value.iterative_check_for_none()

    def iterative_add_output_path(self, output_path: str):
        self.output_path = output_path
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                value.iterative_add_output_path(output_path)




class Parser(argparse.ArgumentParser):
    """Parser class to get config retrieve config file argument"""

    def __init__(self):
        super().__init__()
        self.add_argument("--config", type=str, default=None)
