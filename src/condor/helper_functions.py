import copy
import os
from typing import List, Union, Dict

import yaml

from src.core.utils import get_date_time_tag


def strip_variable(value) -> str:
    if not isinstance(value, float) and '.' in str(value):
        value = value.split('.')[-2]
    if isinstance(value, float):
        value = f'{value:.0e}'
    value = str(value)
    value = os.path.basename(value)
    if value.startswith('"'):
        value = value[1:]
    if value.endswith('"'):
        value = value[:-1]
    return value


def get_variable_name(variable_name: str) -> str:
    #  '[config][0][name]' -> 'name'
    return strip_variable(variable_name.split(']')[-2].split('[')[-1])


def create_configs(base_config: Union[dict, str], adjustments: Dict[str, list]) -> List[str]:
    if isinstance(base_config, str):
        with open(base_config, 'r') as f:
            base_config = yaml.load(f, Loader=yaml.FullLoader)

    # assert each adjusting variable name comes with an equal number of values
    variable_value_lengths = [len(variable_values) for variable_values in adjustments.values()]
    assert min(variable_value_lengths) == max(variable_value_lengths)

    configs = []
    for value_index in range(variable_value_lengths[0]):
        new_config = copy.deepcopy(base_config)
        # loop over variable names to adjust new config
        for variable_name in adjustments.keys():
            value = adjustments[variable_name][value_index]
            exec(f'new_config{variable_name} = value')
        # use first variable_name to define config_name
        variable_name = list(adjustments.keys())[0]
        value = adjustments[variable_name][value_index]
        config_path = os.path.join(base_config['output_path'], 'configs',
                                   f'{get_date_time_tag()}_{get_variable_name(variable_name)}_'
                                   f'{strip_variable(value)}.yml')
        os.makedirs(os.path.join(base_config['output_path'], 'configs'), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(new_config, f)
        configs.append(config_path)
    return configs
