#!/usr/bin/python3.7
from dataclasses import dataclass

import torch
from torch import nn, optim
from dataclasses_json import dataclass_json

from src.ai.evaluator import EvaluatorConfig, Evaluator
from src.ai.model import Model
from src.core.logger import get_logger, cprint

"""Given model, config, data_loader, trains a model and logs relevant training information

If later more complex algorithms for RL are to be implemented, they should inherent from here.
Allows combination of outputs as weighted sum in one big backward pass.
"""


@dataclass_json
@dataclass
class TrainerConfig(EvaluatorConfig):
    batch_size: int = None
    optimizer: str = 'SGD'
    output_weights: list = None  # for each model output specify impact amount of loss in optimizer step
    learning_rate: float = 0.01

    def __post_init__(self):
        if self.output_weights is not None:
            assert len(self.output_weights) == len(self.data_loader_config.outputs)
        else:
            self.output_weights = [1./len(self.data_loader_config.outputs)] * len(self.data_loader_config.outputs)


class Trainer(Evaluator):

    def __init__(self, config: TrainerConfig, model: Model):
        super().__init__(config, model, quiet=True)
        self._logger = get_logger(name=__name__,
                                  output_path=config.output_path,
                                  quite=False)
        cprint(f'Started.', self._logger)
        self._optimizer = eval(f'torch.optim.{self._config.optimizer}(params=self._model.get_parameters(),'
                               f'lr=self._config.learning_rate)')

    def train(self, epoch: int = -1):
        self._optimizer.zero_grad()
        for batch in sample_shuffled_batch(self._dataset):  # a batch is of type Run
            model_outputs = self._model.forward(list(batch.inputs.values()))
            total_loss = torch.Tensor([0])
            for output_index, output in enumerate(model_outputs):
                targets = batch.outputs.values()[output_index]
                loss = self._criterion(output, targets)
                cprint(f'{batch.outputs.keys()[output_index]}: {loss} {self._config.criterion}.')
                total_loss += self._config.output_weights[output_index] * loss.mean()
            total_loss.backward()  # calculate gradients
            self._optimizer.step()  # apply gradients according to optimizer
        self._model.save_to_checkpoint(tag=str(epoch) if epoch != -1 else '')
