#!/usr/bin/python3.7
from dataclasses import dataclass

from dataclasses_json import dataclass_json

from src.ai.evaluator import EvaluatorConfig, Evaluator
from src.ai.model import Model
from src.core.config_loader import Config
from src.core.logger import get_logger, cprint
from src.data.dataset_loader import DataLoaderConfig, DataLoader

"""Given model, config, data_loader, trains a model and logs relevant training information

If later more complex algorithms for RL are to be implemented, they should inhered from here.
"""


@dataclass_json
@dataclass
class TrainerConfig(EvaluatorConfig):
    batch_size: int = None


class Trainer(Evaluator):

    def __init__(self, config: TrainerConfig, model: Model):
        super().__init__(config, model, quiet=True)
        self._logger = get_logger(name=__name__,
                                  output_path=config.output_path,
                                  quite=False)
        cprint(f'Started.', self._logger)

    def train(self):
        pass
