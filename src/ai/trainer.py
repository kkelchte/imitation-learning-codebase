#!/usr/bin/python3.7

from dataclasses import dataclass
from typing import Union

from dataclasses_json import dataclass_json
from tqdm import tqdm
import torch  # Don't remove

from src.ai.base_net import BaseNet
from src.ai.evaluator import EvaluatorConfig, Evaluator
from src.ai.utils import data_to_tensor
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
    ppo_epsilon: Union[str, float] = "default"
    kl_target: Union[str, float] = "default"
    max_actor_training_iterations: Union[str, int] = "default"
    max_critic_training_iterations: Union[str, int] = "default"

    def __post_init__(self):
        # add options in post_init so they are easy to find
        assert self.phi_key in ["default", "gae", "reward-to-go", "return", "value-baseline"]


class Trainer(Evaluator):

    def __init__(self, config: TrainerConfig, network: BaseNet, super_init: bool = False):
        # use super if this called from sub class.
        super().__init__(config, network, quiet=True)

        if not super_init:
            self._logger = get_logger(name=get_filename_without_extension(__file__),
                                      output_path=config.output_path,
                                      quiet=True)
            cprint(f'Started.', self._logger)
            self._optimizer = eval(f'torch.optim.{self._config.optimizer}')(params=self._net.parameters(),
                                                                            lr=self._config.learning_rate)

    def _save_checkpoint(self, epoch: int = -1):
        if epoch != -1:
            if epoch % self._config.save_checkpoint_every_n == 0:
                self._net.save_to_checkpoint(tag=f'{epoch:08}' if epoch != -1 else '')
        else:
            self._net.save_to_checkpoint()

    def train(self, epoch: int = -1, writer=None) -> str:
        self.put_model_on_device()
        total_error = []
#        for batch in tqdm(self.data_loader.sample_shuffled_batch(), ascii=True, desc='train'):
        for batch in self.data_loader.sample_shuffled_batch():
            self._optimizer.zero_grad()
            predictions = self._net.forward(batch.observations, train=True)
            targets = data_to_tensor(batch.actions).type(self._net.dtype).to(self._device)
            loss = self._criterion(predictions, targets).mean()
            loss.backward()  # calculate gradients
            self._optimizer.step()  # apply gradients according to optimizer
            self._net.global_step += 1
            total_error.append(loss.cpu().detach())
        self.put_model_back_to_original_device()
        self._save_checkpoint(epoch)

        error_distribution = Distribution(total_error)
        if writer is not None:
            writer.set_step(self._net.global_step)
            writer.write_distribution(error_distribution, 'training')
        return f' training {self._config.criterion} {error_distribution.mean: 0.3e} [{error_distribution.std:0.2e}]'
