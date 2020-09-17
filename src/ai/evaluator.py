#!/usr/bin/python3
from typing import Tuple

from dataclasses import dataclass
import torch
from torch import nn
import numpy as np
from dataclasses_json import dataclass_json
from tqdm import tqdm

from src.ai.base_net import BaseNet
from src.ai.utils import data_to_tensor
from src.core.config_loader import Config
from src.core.data_types import Distribution
from src.core.logger import get_logger, cprint
from src.core.tensorboard_wrapper import TensorboardWrapper
from src.core.utils import get_filename_without_extension, save_output_plots, create_output_video
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
    evaluate_extensive: bool = False
    store_output_on_tensorboard: bool = False


class Evaluator:

    def __init__(self, config: EvaluatorConfig, network: BaseNet, quiet: bool = False):
        self._config = config
        self._net = network
        self.data_loader = DataLoader(config=self._config.data_loader_config)

        if not quiet:
            self._logger = get_logger(name=get_filename_without_extension(__file__),
                                      output_path=config.output_path,
                                      quiet=True) if type(self) == Evaluator else None
            cprint(f'Started.', self._logger)

        self._device = torch.device(
            "cuda" if self._config.device in ['gpu', 'cuda'] and torch.cuda.is_available() else "cpu"
        )
        self._criterion = eval(f'nn.{self._config.criterion}(reduction=\'none\').to(self._device)')
        self._lowest_validation_loss = None
        self.data_loader.load_dataset()

        self._minimum_error = float(10**6)
        self._original_model_device = self._net.get_device()

    def put_model_on_device(self, device: str = None):
        self._original_model_device = self._net.get_device()
        self._net.set_device(torch.device(self._config.device) if device is None else torch.device(device))

    def put_model_back_to_original_device(self):
        self._net.set_device(self._original_model_device)

    def evaluate(self, epoch: int = -1, writer=None) -> Tuple[str, bool]:
        self.put_model_on_device()
        total_error = []
#        for batch in tqdm(self.data_loader.get_data_batch(), ascii=True, desc='evaluate'):
        for batch in self.data_loader.get_data_batch():
            predictions = self._net.forward(batch.observations, train=False)
            targets = data_to_tensor(batch.actions).type(self._net.dtype).to(self._device)
            error = self._criterion(predictions,
                                    targets).mean()
            total_error.append(error)
        error_distribution = Distribution(total_error)
        self.put_model_back_to_original_device()
        if writer is not None:
            writer.write_distribution(error_distribution, 'validation')
            if self._config.store_output_on_tensorboard and epoch % 30 == 0:
                writer.write_output_image(predictions, 'validation/predictions')
                writer.write_output_image(targets, 'validation/targets')
                writer.write_output_image(torch.stack(batch.observations), 'validation/inputs')

        msg = f' validation {self._config.criterion} {error_distribution.mean: 0.3e} [{error_distribution.std:0.2e}]'

        best_checkpoint = False
        if self._lowest_validation_loss is None or error_distribution.mean < self._lowest_validation_loss:
            self._lowest_validation_loss = error_distribution.mean
            best_checkpoint = True
        return msg, best_checkpoint

    def evaluate_extensive(self) -> None:
        """
        Extra offline evaluation methods for an extensive evaluation at the end of training
        :return: None
        """
        self.put_model_on_device('cpu')
        dataset = self.data_loader.get_dataset()
        predictions = self._net.forward(dataset.observations, train=False).detach().cpu()
        error = predictions - torch.stack(dataset.actions)
        self.put_model_back_to_original_device()

        save_output_plots(output_dir=self._config.output_path,
                          data={'expert': np.stack(dataset.actions),
                                'network': predictions.numpy(),
                                'difference': error.numpy()})
        create_output_video(output_dir=self._config.output_path,
                            observations=dataset.observations,
                            actions={'expert': np.stack(dataset.actions),
                                     'network': predictions.numpy()})

    def remove(self):
        self.data_loader.remove()
        [h.close() for h in self._logger.handlers]
