#!/usr/bin/python3.7

from dataclasses import dataclass
import torch
from torch import nn
from dataclasses_json import dataclass_json
from tqdm import tqdm

from src.ai.model import Model
from src.core.config_loader import Config
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension
from src.data.dataset_loader import DataLoaderConfig, DataLoader

"""Given model, config, data_loader, evaluates a model and logs relevant training information

Depends on ai/architectures, data/data_loader, core/logger
"""


@dataclass_json
@dataclass
class EvaluatorConfig(Config):
    data_loader_config: DataLoaderConfig = None
    criterion: str = 'MSELoss'
    device: str = 'cpu'


class Evaluator:

    def __init__(self, config: EvaluatorConfig, model: Model, quiet: bool = False):
        self._config = config
        self._model = model
        self._data_loader = DataLoader(config=self._config.data_loader_config)

        self._logger = get_logger(name=get_filename_without_extension(__file__),
                                  output_path=config.output_path,
                                  quite=False) if type(self) == Evaluator else None
        if not quiet:
            cprint(f'Started.', self._logger)

        self._device = torch.device(self._config.device)
        self._criterion = eval(f'nn.{self._config.criterion}(reduction=\'none\').to(self._device)')

        self._data_loader.load_dataset(input_sizes=self._model.get_input_sizes(),
                                       output_sizes=self._model.get_output_sizes())

        self._minimum_error = float(10**6)
        self._original_model_device = self._model.get_device()

    def put_model_on_device(self):
        self._original_model_device = self._model.get_device()
        self._model.set_device(self._config.device)

    def put_model_back_to_original_device(self):
        self._model.set_device(self._original_model_device)

    def evaluate(self, save_checkpoints: bool = False) -> float:
        self.put_model_on_device()
        total_error = []
        for run in tqdm(self._data_loader.get_data(), ascii=True, desc='evaluate'):
            model_outputs = self._model.forward(inputs=run.get_input())
            for output_index, output in enumerate(model_outputs):
                targets = run.get_output()[output_index].to(self._config.device)
                error = self._criterion(output, targets).mean()
                total_error.append(error)
                cprint(f'{list(run.outputs.keys())[output_index]}: {error} {self._config.criterion}.', self._logger)
        average_error = float(torch.Tensor(total_error).mean())
        if save_checkpoints and average_error < self._minimum_error:
            self._model.save_to_checkpoint(tag='best')
            self._minimum_error = average_error
        self.put_model_back_to_original_device()
        return average_error
