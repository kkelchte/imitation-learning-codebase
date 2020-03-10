#!/usr/bin/python3.7
from dataclasses import dataclass
from typing import List

import torch
from torch import nn, optim
from dataclasses_json import dataclass_json
from tqdm import tqdm

from src.ai.evaluator import EvaluatorConfig, Evaluator
from src.ai.model import Model
from src.core.config_loader import Config
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension

"""Given model, config, data_loader, trains a model and logs relevant training information

If later more complex algorithms for RL are to be implemented, they should inherent from here.
Allows combination of outputs as weighted sum in one big backward pass.
"""


@dataclass_json
@dataclass
class TrainerConfig(EvaluatorConfig):
    optimizer: str = 'SGD'
    output_weights: List[float] = None  # for each model output specify impact amount of loss in optimizer step
    learning_rate: float = 0.01
    save_checkpoint_every_n: int = 10
    batch_size: int = None

    def post_init(self):
        if self.output_weights is not None:
            assert len(self.output_weights) == len(self.data_loader_config.outputs)
        else:
            self.output_weights = [1./len(self.data_loader_config.outputs)] * len(self.data_loader_config.outputs)
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                value.post_init()


class Trainer(Evaluator):

    def __init__(self, config: TrainerConfig, model: Model):
        super().__init__(config, model, quiet=True)
        self._logger = get_logger(name=get_filename_without_extension(__file__),
                                  output_path=config.output_path,
                                  quite=False)
        cprint(f'Started.', self._logger)
        self._optimizer = eval(f'torch.optim.{self._config.optimizer}(params=self._model.get_parameters(),'
                               f'lr=self._config.learning_rate)')

    def train(self, epoch: int = -1) -> float:
        self.put_model_on_device()
        self._optimizer.zero_grad()
        total_error = []
        for batch in tqdm(self._data_loader.sample_shuffled_batch(), ascii=True, desc='train'):  # type(batch) == Run
            model_outputs = self._model.forward(batch.get_input())
            total_loss = torch.Tensor([0]).to(self._device)
            for output_index, output in enumerate(model_outputs):
                targets = batch.get_output()[output_index].to(self._device)
                loss = self._criterion(output, targets).mean()
                cprint(f'{list(batch.outputs.keys())[output_index]}: {loss} {self._config.criterion}.', self._logger)
                total_loss += self._config.output_weights[output_index] * loss
            total_loss.backward()  # calculate gradients
            self._optimizer.step()  # apply gradients according to optimizer
            total_error.append(total_loss.cpu().detach())

        if epoch != -1:
            if epoch % self._config.save_checkpoint_every_n == 0:
                self._model.save_to_checkpoint(tag=f'{epoch:08}' if epoch != -1 else '')
        else:
            self._model.save_to_checkpoint()
        self.put_model_back_to_original_device()
        return float(torch.Tensor(total_error).mean())
