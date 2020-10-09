#!/usr/bin/python3.8
import torch
from torch import nn
import numpy as np

from src.ai.base_net import BaseNet
from src.ai.trainer import TrainerConfig, Trainer
from src.ai.utils import get_reward_to_go, get_checksum_network_parameters, data_to_tensor
from src.core.data_types import Distribution, Dataset
from src.core.logger import get_logger, cprint
from src.core.tensorboard_wrapper import TensorboardWrapper
from src.core.utils import get_filename_without_extension

"""Given model, config, data_loader, trains a model and logs relevant training information

If later more complex algorithms for RL are to be implemented, they should inherent from here.
Allows combination of outputs as weighted sum in one big backward pass.
"""


class DeepSupervisionConfidence(Trainer):

    def __init__(self, config: TrainerConfig, network: BaseNet):
        super().__init__(config, network, quiet=True)
        self._logger = get_logger(name=get_filename_without_extension(__file__),
                                  output_path=config.output_path,
                                  quiet=True)
        self._optimizer = eval(f'torch.optim.{self._config.optimizer}')(params=self._net.parameters(),
                                                                        lr=self._config.learning_rate,
                                                                        weight_decay=self._config.weight_decay)
        lambda_function = lambda f: 1 - f / self._config.scheduler_config.number_of_epochs
        self._scheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer,
                                                            lr_lambda=lambda_function) \
            if self._config.scheduler_config is not None else None
        cprint(f'Started.', self._logger)

    def train(self, epoch: int = -1, writer=None) -> str:
        self.put_model_on_device()
        total_error = []
        for batch in self.data_loader.sample_shuffled_batch():
            self._optimizer.zero_grad()
            targets = data_to_tensor(batch.actions).type(self._net.dtype).to(self._device)
            probabilities, confidences, final_probability = self._net.forward_with_all_outputs(batch.observations,
                                                                                               train=True)
            loss = self._criterion(final_probability, targets).mean()
            for index, prob in enumerate(probabilities):
                loss += (confidences[index] * (prob - targets)**2).mean()
            loss.sum().backward()
            if self._config.gradient_clip_norm != -1:
                nn.utils.clip_grad_norm_(self._net.parameters(),
                                         self._config.gradient_clip_norm)
            self._optimizer.step()
            self._net.global_step += 1
            total_error.append(loss.cpu().detach())
        self.put_model_back_to_original_device()

        error_distribution = Distribution(total_error)
        if writer is not None:
            writer.set_step(self._net.global_step)
            writer.write_distribution(error_distribution, 'training')
            if self._config.store_output_on_tensorboard and epoch % 30 == 0:
                for index, prob in enumerate(probabilities):
                    writer.write_output_image(prob, f'training/predictions_{index}')
                    writer.write_output_image(confidences[index], f'training/confidences_{index}')
                writer.write_output_image(final_probability, f'training/final_prediction')
                writer.write_output_image(targets, 'training/targets')
                writer.write_output_image(torch.stack(batch.observations), 'training/inputs')
        return f' training {self._config.criterion} {error_distribution.mean: 0.3e} [{error_distribution.std:0.2e}]'
