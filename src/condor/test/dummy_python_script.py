#!/usr/python3.7
import shlex
import subprocess
import sys
from dataclasses import dataclass

from dataclasses_json import dataclass_json

from src.core.config_loader import Parser, Config
from src.core.utils import *


@dataclass_json
@dataclass
class DummyConfig(Config):
    output_path: str = None


config_file = Parser().parse_args().config
if config_file is not None:
    config = DummyConfig().create(config_file=config_file)
    for i in range(3):
        print(f'saving file {i} in {config.output_path}')
        with open(os.path.join(config.output_path, f'dummy_file_{i}'), 'w') as f:
            try:
                msg = os.environ['_CONDOR_JOB_AD']
            except:
                msg = 'failed to read _CONDOR_JOB_AD'
            f.write(msg)
    nested_path = os.path.join(config.output_path, 'nested_dir_1', 'nested_dir_2')
    os.makedirs(nested_path, exist_ok=True)
    with open(os.path.join(nested_path, 'already_existing_file'), 'w') as f:
        f.write('overwritten\noverwritten\noverwritten')
    with open(os.path.join(nested_path, 'new_file'), 'w') as f:
        f.write('fresh')

subprocess.call(shlex.split("printenv"))


print('well done!')
sys.exit(2)
