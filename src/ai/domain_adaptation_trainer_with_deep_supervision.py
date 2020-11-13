#!/usr/bin/python3.8
import torch
from torch import nn
import numpy as np

from src.ai.base_net import BaseNet
from src.ai.domain_adaptation_trainer import DomainAdaptationTrainer
from src.ai.trainer import TrainerConfig, Trainer
from src.ai.losses import *
from src.ai.deep_supervision import DeepSupervision
from src.ai.utils import get_reward_to_go, get_checksum_network_parameters, data_to_tensor
from src.core.data_types import Distribution, Dataset
from src.core.logger import get_logger, cprint
from src.core.tensorboard_wrapper import TensorboardWrapper
from src.core.utils import get_filename_without_extension
from src.data.data_loader import DataLoader

"""Given model, config, data_loader, trains a model and logs relevant training information
Adds extra data loader for target data.
Use epsilon to specify balance between (1 - epsilon) * L_task + epsilon * L_domain_adaptation
"""


class DeepSupervisedDomainAdaptationTrainer(DeepSupervision, DomainAdaptationTrainer):

    def __init__(self, config: TrainerConfig, network: BaseNet, quiet: bool = False):
        super(DeepSupervisedDomainAdaptationTrainer, self).__init__(config, network, quiet=True)
        if not quiet:
            self._logger = get_logger(name=get_filename_without_extension(__file__),
                                      output_path=config.output_path,
                                      quiet=False)
            cprint(f'Started.', self._logger)

    def train(self, epoch: int = -1, writer=None) -> str:
        self.put_model_on_device()
        total_error = []
        for source_batch, target_batch in zip(self.data_loader.sample_shuffled_batch(),
                                              self.target_data_loader.sample_shuffled_batch()):
            self._optimizer.zero_grad()
            targets = data_to_tensor(source_batch.actions).type(self._net.dtype).to(self._device)

            # deep supervision loss
            probabilities = self._net.forward_with_all_outputs(source_batch.observations, train=True)
            loss = self._criterion(probabilities[-1], targets).mean()
            for index, prob in enumerate(probabilities[:-1]):
                loss += self._criterion(prob, targets).mean()
            loss *= (1 - self.epsilon)

            # add domain adaptation loss
            loss += self.epsilon * self._domain_adaptation_criterion(self._net.get_features(source_batch.observations),
                                                                     self._net.get_features(target_batch.observations))

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
            if self._config.store_output_on_tensorboard and epoch % 30 == 0:
                writer.write_output_image(probabilities[-1], 'source/predictions')
                writer.write_output_image(targets, 'source/targets')
                writer.write_output_image(torch.stack(source_batch.observations), 'source/inputs')
                writer.write_output_image(self._net.forward(target_batch.observations, train=True),
                                          'target/predictions')
                writer.write_output_image(torch.stack(target_batch.observations), 'target/inputs')

        return f' training {self._config.criterion} {error_distribution.mean: 0.3e} [{error_distribution.std:0.2e}]'
