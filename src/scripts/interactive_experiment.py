
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from src.core.config_loader import Config, Parser
from src.core.logger import get_logger, cprint
from src.sim.common.environment_runner import EnvironmentRunnerConfig, EnvironmentRunner
from src.data.dataset_saver import DataSaverConfig, DataSaver

"""Script for collecting dataset / evaluating a model in simulated or real environment.

Script starts environment runner with dataset_saver object to store the episodes in a dataset.
"""


@dataclass_json
@dataclass
class InteractiveExperimentConfig(Config):
    runner_config: EnvironmentRunnerConfig = None
    data_saver_config: DataSaverConfig = None


class InteractiveExperiment:

    def __init__(self, config: InteractiveExperimentConfig):
        self._logger = get_logger(name=__name__,
                                  output_path=config.output_path,
                                  quite=False)
        cprint(f'Started.', self._logger)
        self._data_saver = DataSaver(config=config.data_saver_config)
        self._environment_runner = EnvironmentRunner(config=config.runner_config,
                                                     data_saver=self._data_saver)

    def run(self):
        self._environment_runner.run()

    def shutdown(self):
        self._environment_runner.shutdown()


if __name__ == "__main__":
    config_file = Parser().parse_args().config
    experiment_config = InteractiveExperimentConfig().create(config_file=config_file)
    experiment = InteractiveExperiment(experiment_config)
    experiment.run()

