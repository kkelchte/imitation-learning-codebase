#!/usr/bin/python3

from dataclasses import dataclass
from typing import Union, Optional

from dataclasses_json import dataclass_json
from torch import nn
from tqdm import tqdm
import torch  # Don't remove

from src.ai.base_net import BaseNet
from src.ai.evaluator import EvaluatorConfig, Evaluator
from src.ai.utils import data_to_tensor
from src.core.config_loader import Config
from src.core.data_types import Distribution
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension
from src.data.data_loader import DataLoaderConfig

"""Given model, config, data_loader, trains a model and logs relevant training information

If later more complex algorithms for RL are to be implemented, they should inherent from here.
Allows combination of outputs as weighted sum in one big backward pass.
"""


@dataclass_json
@dataclass
class SchedulerConfig(Config):
    number_of_epochs: int = None


@dataclass_json
@dataclass
class TrainerConfig(EvaluatorConfig):
    optimizer: str = 'SGD'
    learning_rate: float = 0.01
    actor_learning_rate: float = -1
    critic_learning_rate: float = -1
    scheduler_config: Optional[SchedulerConfig] = None
    entropy_coefficient: float = 0
    weight_decay: float = 0
    confidence_weight: float = 0
    gradient_clip_norm: float = -1
    factory_key: str = "BASE"
    phi_key: str = "default"
    discount: Union[str, float] = "default"
    gae_lambda: Union[str, float] = "default"
    epsilon: Union[str, float] = "default"
    use_kl_stop: bool = False
    kl_target: Union[str, float] = "default"
    max_actor_training_iterations: Union[str, int] = "default"
    max_critic_training_iterations: Union[str, int] = "default"
    add_KL_divergence_loss: bool = False
    target_data_loader_config: Optional[DataLoaderConfig] = None
    domain_adaptation_criterion: str = "default"

    def __post_init__(self):
        # add options in post_init so they are easy to find
        if self.scheduler_config is None:
            del self.scheduler_config
        if self.target_data_loader_config is None:
            del self.target_data_loader_config
        assert self.phi_key in ["default", "gae", "reward-to-go", "return", "value-baseline"]


class Trainer(Evaluator):

    def __init__(self, config: TrainerConfig, network: BaseNet, quiet: bool = False):
        # use super if this called from sub class.
        super().__init__(config, network, quiet=True)

        if not quiet:
            self._optimizer = eval(f'torch.optim.{self._config.optimizer}')(params=self._net.parameters(),
                                                                            lr=self._config.learning_rate,
                                                                            weight_decay=self._config.weight_decay)

            lambda_function = lambda f: 1 - f / self._config.scheduler_config.number_of_epochs
            self._scheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer, lr_lambda=lambda_function) \
                if self._config.scheduler_config is not None else None
            self._logger = get_logger(name=get_filename_without_extension(__file__),
                                      output_path=config.output_path,
                                      quiet=False)
            cprint(f'Started.', self._logger)

    def train(self, epoch: int = -1, writer=None) -> str:
        self.put_model_on_device()
        total_error = []
        #        for batch in tqdm(self.data_loader.sample_shuffled_batch(), ascii=True, desc='train'):
        for batch in self.data_loader.sample_shuffled_batch():
            self._optimizer.zero_grad()
            targets = data_to_tensor(batch.actions).type(self._net.dtype).to(self._device)
            if self._config.add_KL_divergence_loss:
                predictions, mean, std = self._net.forward_with_distribution(batch.observations, train=True)
            else:
                predictions = self._net.forward(batch.observations, train=True)

            loss = self._criterion(predictions, targets).mean()
            if self._config.add_KL_divergence_loss:
                # https://arxiv.org/pdf/1312.6114.pdf
                KL_loss = -0.5 * torch.sum(1 + std.pow(2).log() - mean.pow(2) - std.pow(2))
                loss += KL_loss

            loss.backward()
            if self._config.gradient_clip_norm != -1:
                nn.utils.clip_grad_norm_(self._net.parameters(),
                                         self._config.gradient_clip_norm)
            self._optimizer.step()
            self._net.global_step += 1
            total_error.append(loss.cpu().detach())
        self.put_model_back_to_original_device()

        if self._scheduler is not None:
            self._scheduler.step()

        error_distribution = Distribution(total_error)
        if writer is not None:
            writer.set_step(self._net.global_step)
            writer.write_distribution(error_distribution, 'training')
            if self._config.add_KL_divergence_loss:
                writer.write_scalar(KL_loss, 'KL_divergence')
            if self._config.store_output_on_tensorboard and epoch % 30 == 0:
                writer.write_output_image(predictions, 'training/predictions')
                writer.write_output_image(targets, 'training/targets')
                writer.write_output_image(torch.stack(batch.observations), 'training/inputs')
        return f' training {self._config.criterion} {error_distribution.mean: 0.3e} [{error_distribution.std:0.2e}]'

    def get_checkpoint(self) -> dict:
        """
        makes a checkpoint for the trainers field.
        :return: dictionary with optimizers state_dict and schedulers state dict
        """
        checkpoint = {'optimizer_state_dict': self._optimizer.state_dict()}
        if self._scheduler is not None:
            checkpoint['scheduler_state_dict'] = self._scheduler.state_dict()
        return checkpoint

    def load_checkpoint(self, checkpoint: dict) -> None:
        """
        Load optimizers and schedulers state_dict
        :return: None
        """
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self._scheduler is not None:
            self._scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self._device.type != 'cpu':
            for state in self._optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
