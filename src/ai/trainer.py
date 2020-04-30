#!/usr/bin/python3.7
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn, optim
from dataclasses_json import dataclass_json
from tqdm import tqdm

from src.ai.base_net import BaseNet
from src.ai.evaluator import EvaluatorConfig, Evaluator
from src.core.config_loader import Config
from src.core.data_types import Distribution
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
    learning_rate: float = 0.01
    save_checkpoint_every_n: int = 10


class Trainer(Evaluator):

    def __init__(self, config: TrainerConfig, network: BaseNet):
        super().__init__(config, network, quiet=True)
        self._logger = get_logger(name=get_filename_without_extension(__file__),
                                  output_path=config.output_path,
                                  quite=False)
        cprint(f'Started.', self._logger)
        self._optimizer = eval(f'torch.optim.{self._config.optimizer}')(params=self._net.parameters(),
                                                                        lr=self._config.learning_rate)

    def train(self, epoch: int = -1) -> Distribution:
        self.put_model_on_device()
        self._optimizer.zero_grad()
        total_error = []
        for batch in tqdm(self._data_loader.sample_shuffled_batch(), ascii=True, desc='train'):  # type(batch) == Run
            predictions = self._net.forward(batch.observations, train=True)
            targets = torch.as_tensor(batch.actions, dtype=self._net.dtype).to(self._device)
            loss = self._criterion(predictions, targets).mean()
            loss.backward()  # calculate gradients
            self._optimizer.step()  # apply gradients according to optimizer
            self._net.global_step += 1
            total_error.append(loss.cpu().detach())

        if epoch != -1:
            if epoch % self._config.save_checkpoint_every_n == 0:
                self._net.save_to_checkpoint(tag=f'{epoch:08}' if epoch != -1 else '')
        else:
            self._net.save_to_checkpoint()
        self.put_model_back_to_original_device()
        return Distribution(
            mean=float(torch.as_tensor(total_error).mean()),
            std=float(torch.as_tensor(total_error).std())
        )
