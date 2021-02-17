import os
from glob import glob

import yaml

from src.core.utils import get_to_root_dir
from src.scripts.data_cleaning import DataCleaningConfig
from src.scripts.experiment import ExperimentConfig

CONFIG_DICT = {}
CONFIG_FILENAME = "rl_tracking_ppo.yml"
CONFIG_FILE = f"src/scripts/config/{CONFIG_FILENAME}"


def write_config(file_name: str, config_dict: dict) -> None:
    with open(file_name, 'w') as f:
        yaml.dump(config_dict, f)
    print(f"write {file_name} OK")


def read_config(file_name: str) -> dict:
    with open(file_name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(f"read {file_name} OK")
    return config


def validate_config(filename) -> None:
    config_dict = read_config(filename)
    if not 'data_cleaning' in filename:
        ExperimentConfig().create(config_dict=config_dict, store=False)
    else:
        DataCleaningConfig().create(config_dict=config_dict, store=False)
    print("validation OK")


if __name__ == "__main__":
    get_to_root_dir()
    for f in glob(f'./src/scripts/config/*.yml'):
        print(f)
        validate_config(f)
        # write_config(destination, base_config)
