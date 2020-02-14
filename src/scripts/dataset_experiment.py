#!/usr/bin/python3.7
import os
from dataclasses import dataclass

from dataclasses_json import dataclass_json

from src.ai.evaluator import EvaluatorConfig, Evaluator
from src.ai.model import ModelConfig, Model
from src.ai.trainer import TrainerConfig, Trainer
from src.core.config_loader import Config, Parser
from src.core.logger import get_logger, cprint
from src.core.utils import get_date_time_tag

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

    def iterative_add_output_path(self, output_path: str) -> None:
        # assuming output_path is standard ${experiment_name}
        self.output_path = os.path.join(output_path, 'models',
                                        f'{get_date_time_tag()}_{self.model_config.architecture}')
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                value.iterative_add_output_path(self.output_path)


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
            msg = f'ended epoch: {epoch} / {self._config.number_of_epochs}'
            if self._trainer is not None:
                training_error = self._trainer.train(epoch=epoch)  # include checkpoint saving.
                msg += f' training error: {training_error}'
            if self._evaluator is not None:  # if validation error is minimal then save best checkpoint
                validation_error = self._evaluator.evaluate(save_checkpoints=self._trainer is not None)
                msg += f' validation error: {validation_error}'
            cprint(msg, self._logger)

        cprint(f'Finished.', self._logger)


if __name__ == "__main__":
    config_file = Parser().parse_args().config
    experiment_config = DatasetExperimentConfig().create(config_file=config_file)
    experiment = DatasetExperiment(experiment_config)
    experiment.run()

