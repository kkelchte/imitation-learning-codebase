import os
import yaml
from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Config:

    def create(self, config_dict: dict = {}, config_file: str = ''):
        assert not (config_file and config_dict)
        assert (config_dict or config_file)

        if config_file:
            if not os.path.exists(config_file):
                raise FileNotFoundError('Are you in the code root directory?')
            with open(config_file, 'r') as f:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)

        instant = self.from_dict(config_dict)
        for field in instant.__dict__.values():
            assert field is not None
        return instant
