import os
import argparse

import yaml
from dataclasses import dataclass
from dataclasses_json import dataclass_json


def iterative_check_for_none_values(dictionary: dict):
    for key, value in dictionary.items():
        if value is None:
            raise IOError(f'Found None value in config. \n {key}: {value}')
        if isinstance(value, dict):
            iterative_check_for_none_values(value)


@dataclass_json
@dataclass
class Config:
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
        iterative_check_for_none_values(instant.__dict__)
        return instant


class Parser(argparse.ArgumentParser):
    """Parser class to get config retrieve config file argument"""

    def __init__(self):
        super().__init__()
        self.add_argument("--config", type=str, default=None)
