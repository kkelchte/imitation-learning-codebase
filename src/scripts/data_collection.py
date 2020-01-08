
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from src.core.config_loader import Config, Parser
from src.sim.common.environment import EnvironmentConfig

"""Script for collecting dataset in simulated or real environment.

Script starts environment runner with dataset_saver object to generate a dataset.
"""


@dataclass_json
@dataclass
class DataCollectionConfig(Config):
    output_path: str = None
    environment_config: EnvironmentConfig = None


def main():
    config_file = Parser().parse_args().config
    config = DataCollectionConfig().create(config_file=config_file)
    print(config)


if __name__ == "__main__":
    main()
