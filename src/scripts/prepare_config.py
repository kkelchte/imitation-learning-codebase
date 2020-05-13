import os

import yaml

from src.core.utils import get_to_root_dir
from src.scripts.experiment import ExperimentConfig

CONFIG_DICT = {}
CONFIG_FILENAME = "il_evaluate_interactive_cube_world.yml"
CONFIG_FILE = f"src/scripts/config/{CONFIG_FILENAME}"
get_to_root_dir()


def write_config(file_name: str, config_dict: dict) -> None:
    with open(file_name, 'w') as f:
        yaml.dump(config_dict, f)
    print(f"write {file_name} OK")


def read_config(file_name: str) -> dict:
    with open(file_name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(f"read {file_name} OK")
    return config


def validate_config(config_dict: dict) -> None:
    ExperimentConfig().create(config_dict=config_dict, store=False)
    print("validation OK")


if __name__ == "__main__":
    pass
    config = read_config(f'{os.environ["HOME"]}/experimental_data/cube_world/condor/'
                         f'20-05-13_08-46-59_initialisation_seed_5100/adjusted_config.yml')
    # adjust config if needed here
    validate_config(config)
    # write_config(destination, base_config)
