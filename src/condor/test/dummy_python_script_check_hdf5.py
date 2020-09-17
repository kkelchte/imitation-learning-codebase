#!/usr/python
import os
import sys

import yaml

from src.core.config_loader import Parser
from src.core.utils import *


config_file = Parser().parse_args().config
with open(config_file, 'r') as f:
    config_dict = yaml.load(f, Loader=yaml.FullLoader)


def print_hdf5_files_locations(config: dict):
    for key, value in config:
        if key == 'hdf5_files':
            for v in value:
                print(f'HDF5_FILE {v} {os.system("stat --format %s "+v)}')
                assert os.path.isfile(v)
        if isinstance(value, dict):
            print_hdf5_files_locations(value)


print_hdf5_files_locations(config_dict)
print('well done!')
sys.exit(2)
