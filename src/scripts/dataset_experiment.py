#!/usr/bin/python3.7
from dataclasses import dataclass
from typing import Optional

from dataclasses_json import dataclass_json

from src.ai.evaluator import EvaluatorConfig, Evaluator
from src.ai.model import ModelConfig, Model
from src.ai.trainer import TrainerConfig, Trainer
from src.core.config_loader import Config, Parser
from src.core.logger import get_logger, cprint
from src.data.dataset_loader import DataLoaderConfig
from src.sim.common.environment_runner import EnvironmentRunnerConfig, EnvironmentRunner
from src.data.dataset_saver import DataSaverConfig, DataSaver

"""Script for collecting dataset / evaluating a model in simulated or real environment.

Script starts environment runner with dataset_saver object to store the episodes in a dataset.
"""


@dataclass_json
@dataclass
class DatasetExperimentConfig(Config):
    model_config: ModelConfig = None
    trainer_config: TrainerConfig = None
    evaluator_config: EvaluatorConfig = None
    number_of_epochs: int = 1

    def __post_init__(self):
        # Avoid None value error by deleting irrelevant fields
        if self.trainer_config is None:
            del self.trainer_config
        if self.evaluator_config is None:
            del self.evaluator_config


class DatasetExperiment:

    def __init__(self, config: DatasetExperimentConfig):
        self._config = config
        self._logger = get_logger(name=__name__,
                                  output_path=config.output_path,
                                  quite=False)
        cprint(f'Started.', self._logger)
        self._model = Model(config=self._config.model_config)
        self._trainer = Trainer(config=self._config.trainer_config, model=self._model) \
            if self._config.trainer_config is not None else None
        self._evaluator = Evaluator(config=self._config.evaluator_config, model=self._model) \
            if self._config.evaluator_config is not None else None

    def run(self):
        for epoch in range(self._config.number_of_epochs):
            cprint(f'epoch: {epoch} / {self._config.number_of_epochs}', self._logger)
            if self._trainer is not None:
                self._trainer.train()
            if self._evaluator is not None:
                self._evaluator.evaluate()
            self._model.save()
        cprint(f'Finished.', self._logger)


if __name__ == "__main__":
    config_file = Parser().parse_args().config
    experiment_config = DatasetExperimentConfig().create(config_file=config_file)
    experiment = DatasetExperiment(experiment_config)
    experiment.run()

