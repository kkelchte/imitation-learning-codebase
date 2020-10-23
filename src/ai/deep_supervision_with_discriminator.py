#!/usr/bin/python3.8
import torch
from torch import nn
import numpy as np

from src.ai.base_net import BaseNet
from src.ai.trainer import TrainerConfig
from src.ai.deep_supervision import DeepSupervision
from src.ai.utils import get_reward_to_go, get_checksum_network_parameters, data_to_tensor
from src.core.data_types import Distribution, Dataset
from src.core.logger import get_logger, cprint
from src.core.tensorboard_wrapper import TensorboardWrapper
from src.core.utils import get_filename_without_extension
from src.data.data_loader import DataLoader

"""Given model, config, data_loader, trains a model and logs relevant training information

If later more complex algorithms for RL are to be implemented, they should inherent from here.
Allows combination of outputs as weighted sum in one big backward pass.
"""


class DeepSupervisionWithDiscriminator(DeepSupervision):

    def __init__(self, config: TrainerConfig, network: BaseNet, quiet: bool = False):
        super().__init__(config, network, quiet=True)
        self._discriminator_optimizer = eval(f'torch.optim.{self._config.optimizer}')(
            params=self._net.discriminator_parameters(),
            lr=self._config.critic_learning_rate if self._config.critic_learning_rate != -1
            else self._config.learning_rate,
            weight_decay=self._config.weight_decay)

        self.discriminator_data_loader = DataLoader(config=self._config.discriminator_data_loader_config)
        self.discriminator_data_loader.load_dataset()

        if not quiet:
            self._logger = get_logger(name=get_filename_without_extension(__file__),
                                      output_path=config.output_path,
                                      quiet=False)
            cprint(f'Started.', self._logger)

    def _train_discriminator_network(self, epoch: int = -1, writer=None) -> str:
        total_error = []
        criterion = nn.BCELoss()
        for sim_batch, real_batch in zip(self.data_loader.sample_shuffled_batch(),
                                         self.discriminator_data_loader.sample_shuffled_batch()):
            self._discriminator_optimizer.zero_grad()
            sim_predictions = torch.cat(self._net.forward_with_all_outputs(sim_batch.observations, train=False))
            real_predictions = torch.cat(self._net.forward_with_all_outputs(real_batch.observations, train=False))
            targets = torch.as_tensor([*[0] * len(sim_predictions), *[1] * len(real_predictions)])\
                .type(self._net.dtype).to(self._device)
            outputs = self._net.discriminate(torch.cat([sim_predictions, real_predictions]), train=True)
            loss = criterion(outputs, targets)
            loss.mean().backward()
            if self._config.gradient_clip_norm != -1:
                nn.utils.clip_grad_norm_(self._net.discriminator_parameters(),
                                         self._config.gradient_clip_norm)
            self._discriminator_optimizer.step()
            total_error.append(loss.cpu().detach())

        error_distribution = Distribution(total_error)
        if writer is not None:
            writer.set_step(self._net.global_step)
            writer.write_distribution(error_distribution, 'discriminator')
        return f' train discriminator network BCE {error_distribution.mean: 0.3e}'

    def _train_main_network(self, epoch: int = -1, writer=None) -> str:
        total_error = []
        discriminator_output = []
        for sim_batch, real_batch in zip(self.data_loader.sample_shuffled_batch(),
                                         self.discriminator_data_loader.sample_shuffled_batch()):
            self._optimizer.zero_grad()
            # normal deep supervision loss
            targets = data_to_tensor(sim_batch.actions).type(self._net.dtype).to(self._device)
            probabilities = self._net.forward_with_all_outputs(sim_batch.observations, train=True)
            loss = self._criterion(probabilities[-1], targets).mean()
            for index, prob in enumerate(probabilities[:-1]):
                loss += self._criterion(prob, targets).mean()
            loss.mean().backward()
            total_error.append(loss.mean().cpu().detach())
            # adversarial loss on discriminator data
            discriminator_probabilities = self._net.forward_with_all_outputs(real_batch.observations, train=True)
            discriminator_loss = self._net.discriminate(torch.stack(discriminator_probabilities), train=False)
            discriminator_loss.mean().backward()
            discriminator_output.append(discriminator_loss.mean().cpu().detach())
            # clip gradients
            if self._config.gradient_clip_norm != -1:
                nn.utils.clip_grad_norm_(self._net.parameters(),
                                         self._config.gradient_clip_norm)
            self._optimizer.step()
            self._net.global_step += 1

        error_distribution = Distribution(total_error)
        discriminator_distribution = Distribution(discriminator_output)
        if writer is not None:
            writer.set_step(self._net.global_step)
            writer.write_distribution(error_distribution, 'training_deep_supervision')
            writer.write_distribution(discriminator_distribution, 'training_discriminator')
            if self._config.store_output_on_tensorboard and epoch % 30 == 0:
                for index, prob in enumerate(probabilities):
                    writer.write_output_image(prob, f'training/predictions_{index}')
                writer.write_output_image(targets, 'training/targets')
                writer.write_output_image(torch.stack(sim_batch.observations), 'training/inputs')
            for index, prob in enumerate(discriminator_probabilities):
                writer.write_output_image(prob, f'real/predictions_{index}')
            writer.write_output_image(torch.stack(real_batch.observations), 'real/inputs')
        return f' training {self._config.criterion} {error_distribution.mean: 0.3e} [{error_distribution.std:0.2e}]'

    def train(self, epoch: int = -1, writer=None) -> str:
        self.put_model_on_device()

        # Train deep supervision network on training and discriminated unlabeled real data
        message = self._train_main_network(epoch, writer)

        # Train discriminator network
        message += self._train_discriminator_network(epoch, writer)

        self.put_model_back_to_original_device()
        return message
