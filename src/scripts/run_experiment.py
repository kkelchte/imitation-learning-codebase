from dataclasses import dataclass
from dataclasses_json import dataclass_json

from src.core.config_loader import Config, Parser
from src.sim.common.environment_runner import EnvironmentRunnerConfig, EnvironmentRunner
from src.data.dataset_saver import DataSaverConfig, DataSaver

"""Script for collecting dataset / evaluating a model in simulated or real environment.

Script starts environment runner with dataset_saver object to store the episodes in a dataset.
"""


@dataclass_json
@dataclass
class ExperimentConfig(Config):
    runner_config: EnvironmentRunnerConfig = None
    data_saver_config: DataSaverConfig = None


def main():
    config_file = Parser().parse_args().config
    config = ExperimentConfig().create(config_file=config_file)
    data_saver = DataSaver(config=config.data_saver_config)
    environment_runner = EnvironmentRunner(config=config.runner_config,
                                           data_saver=data_saver)
    environment_runner.run()


if __name__ == "__main__":
    main()
