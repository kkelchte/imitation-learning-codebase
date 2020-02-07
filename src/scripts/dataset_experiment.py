#!/usr/bin/python3.7
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from src.core.config_loader import Config, Parser
from src.core.logger import get_logger
from src.data.dataset_loader import DataLoaderConfig
from src.sim.common.environment_runner import EnvironmentRunnerConfig, EnvironmentRunner
from src.data.dataset_saver import DataSaverConfig, DataSaver

"""Script for collecting dataset / evaluating a model in simulated or real environment.

Script starts environment runner with dataset_saver object to store the episodes in a dataset.
"""


@dataclass_json
@dataclass
class DatasetExperimentConfig(Config):
    data_loader_config: DataLoaderConfig = None


class DatasetExperiment:

    def __init__(self, config: DatasetExperimentConfig):
        logger = get_logger(name=__name__,
                            output_path=config.output_path,
                            quite=False)
        logger.info(f'Started.')

    def run(self):
        pass


if __name__ == "__main__":
    config_file = Parser().parse_args().config
    experiment_config = DatasetExperimentConfig().create(config_file=config_file)
    experiment = DatasetExperiment(experiment_config)
    experiment.run()

