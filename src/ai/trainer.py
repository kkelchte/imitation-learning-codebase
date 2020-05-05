#!/usr/bin/python3.7
from enum import IntEnum

from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
from torch import nn, optim
from dataclasses_json import dataclass_json
from tqdm import tqdm

from src.ai.base_net import BaseNet
from src.ai.evaluator import EvaluatorConfig, Evaluator
from src.ai.utils import data_to_tensor
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
    factory_key: str = "BASE"
    phi_key: str = "default"
    discount: Union[str, float] = "default"
    gae_lambda: Union[str, float] = "default"

    def __post_init__(self):
        # add options in post_init so they are easy to find
        assert self.phi_key in ["default", "gae", "reward-to-go", "return", "value-baseline"]


class Trainer(Evaluator):

    def __init__(self, config: TrainerConfig, network: BaseNet, quiet: bool = False):
        super().__init__(config, network, quiet=True)

        if not quiet:
            self._logger = get_logger(name=get_filename_without_extension(__file__),
                                      output_path=config.output_path,
                                      quite=False)
            cprint(f'Started.', self._logger)
        self._optimizer = eval(f'torch.optim.{self._config.optimizer}')(params=self._net.parameters(),
                                                                        lr=self._config.learning_rate)

    def train(self, epoch: int = -1, writer=None) -> str:
        self.put_model_on_device()
        self._optimizer.zero_grad()
        total_error = []
        for batch in tqdm(self.data_loader.sample_shuffled_batch(), ascii=True, desc='train'):  # type(batch) == Run
            predictions = self._net.forward(batch.observations, train=True)
            targets = data_to_tensor(batch.actions).type(self._net.dtype).to(self._device)
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

        error_distribution = Distribution(
                mean=float(torch.as_tensor(total_error).mean()),
                std=float(torch.as_tensor(total_error).std())
            )
        if writer is not None:
            writer.set_step(self._net.global_step)
            writer.write_distribution(error_distribution, 'training')
        return f'training error mean {error_distribution.mean: 0.3e} [{error_distribution.std: 0.2e}]'
