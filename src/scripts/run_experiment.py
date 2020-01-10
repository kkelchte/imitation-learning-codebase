
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from src.core.config_loader import Config, Parser
from src.sim.common.environment_runner import EnvironmentRunnerConfig, EnvironmentRunner

"""Script for collecting dataset in simulated or real environment.

Script starts environment runner with dataset_saver object to generate a dataset.
"""


@dataclass_json
@dataclass
class ExperimentConfig(Config):
    output_path: str = None
    runner_config: EnvironmentRunnerConfig = None


def main():
    config_file = Parser().parse_args().config
    config = ExperimentConfig().create(config_file=config_file)
    environment_runner = EnvironmentRunner(config=config.runner_config)
    environment_runner.run()


if __name__ == "__main__":
    main()
