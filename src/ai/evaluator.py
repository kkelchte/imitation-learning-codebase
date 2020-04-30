#!/usr/bin/python3.7

from dataclasses import dataclass
import torch
from torch import nn
from dataclasses_json import dataclass_json
from tqdm import tqdm

from src.ai.base_net import BaseNet
from src.ai.utils import data_to_tensor
from src.core.config_loader import Config
from src.core.data_types import Distribution
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension
from src.data.data_loader import DataLoaderConfig, DataLoader

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

    def __init__(self, config: EvaluatorConfig, network: BaseNet, quiet: bool = False):
        self._config = config
        self._net = network
        self._data_loader = DataLoader(config=self._config.data_loader_config)

        self._logger = get_logger(name=get_filename_without_extension(__file__),
                                  output_path=config.output_path,
                                  quite=False) if type(self) == Evaluator else None
        if not quiet:
            cprint(f'Started.', self._logger)

        self._device = torch.device(
            "cuda" if self._config.device in ['gpu', 'cuda'] and torch.cuda.is_available() else "cpu"
        )
        self._criterion = eval(f'nn.{self._config.criterion}(reduction=\'none\').to(self._device)')

        self._data_loader.load_dataset()

        self._minimum_error = float(10**6)
        self._original_model_device = self._net.get_device()

    def put_model_on_device(self):
        self._original_model_device = self._net.get_device()
        self._net.to_device(torch.device(self._config.device))

    def put_model_back_to_original_device(self):
        self._net.to_device(self._original_model_device)

    def evaluate(self, save_checkpoints: bool = False) -> Distribution:
        self.put_model_on_device()
        total_error = []
        for batch in tqdm(self._data_loader.get_data_batch(), ascii=True, desc='evaluate'):
            predictions = self._net.forward(batch.observations)
            error = self._criterion(predictions,
                                    data_to_tensor(batch.actions).type(self._net.dtype).to(self._device)).mean()
            total_error.append(error)
        error_distribution = Distribution(
                mean=float(torch.as_tensor(total_error).mean()),
                std=float(torch.as_tensor(total_error).std())
        )
        if save_checkpoints and error_distribution.mean < self._minimum_error:
            self._net.save_to_checkpoint(tag='best')
            self._minimum_error = error_distribution.mean
        self.put_model_back_to_original_device()
        return error_distribution
