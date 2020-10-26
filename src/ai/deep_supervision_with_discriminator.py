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
        self._config.epsilon = 0.2 if self._config.epsilon == "default" else self._config.epsilon

        del self._optimizer
        self._optimizer = eval(f'torch.optim.{self._config.optimizer}')(params=self._net.deeply_supervised_parameters(),
                                                                        lr=self._config.learning_rate,
                                                                        weight_decay=self._config.weight_decay)
        lambda_function = lambda f: 1 - f / self._config.scheduler_config.number_of_epochs
        del self._scheduler
        self._scheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer, lr_lambda=lambda_function) \
            if self._config.scheduler_config is not None else None

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

    def _train_main_network(self, epoch: int = -1, writer=None) -> str:
        deeply_supervised_error = []
        discriminator_error = []
        for sim_batch, real_batch in zip(self.data_loader.sample_shuffled_batch(),
                                         self.discriminator_data_loader.sample_shuffled_batch()):
            self._optimizer.zero_grad()
            # normal deep supervision loss
            targets = data_to_tensor(sim_batch.actions).type(self._net.dtype).to(self._device)
            probabilities = self._net.forward_with_all_outputs(sim_batch.observations, train=True)
            loss = self._criterion(probabilities[-1], targets).mean()
            for index, prob in enumerate(probabilities[:-1]):
                loss += self._criterion(prob, targets).mean()
            deeply_supervised_error.append(loss.mean().cpu().detach())
            # adversarial loss on discriminator data
            discriminator_probabilities = self._net.forward_with_all_outputs(real_batch.observations, train=True)
            discriminator_loss = self._net.discriminate(torch.stack(discriminator_probabilities), train=False).mean()
            loss += self._config.epsilon * discriminator_loss
            loss.mean().backward()
            discriminator_error.append(discriminator_loss.mean().cpu().detach())
            # clip gradients
            if self._config.gradient_clip_norm != -1:
                nn.utils.clip_grad_norm_(self._net.parameters(),
                                         self._config.gradient_clip_norm)
            self._optimizer.step()
            self._net.global_step += 1

        supervised_error_distribution = Distribution(deeply_supervised_error)
        discriminator_error_distribution = Distribution(discriminator_error)
        if writer is not None:
            writer.set_step(self._net.global_step)
            writer.write_distribution(supervised_error_distribution, 'training_deep_supervision')
            writer.write_distribution(discriminator_error_distribution, 'training_discriminator')
            if self._config.store_output_on_tensorboard and epoch % 30 == 0:
                for index, prob in enumerate(probabilities):
                    writer.write_output_image(prob, f'training/predictions_{index}')
                writer.write_output_image(targets, 'training/targets')
                writer.write_output_image(torch.stack(sim_batch.observations), 'training/inputs')
            for index, prob in enumerate(discriminator_probabilities):
                writer.write_output_image(prob, f'real/predictions_{index}')
            writer.write_output_image(torch.stack(real_batch.observations), 'real/inputs')
        return f' Training: supervision {self._config.criterion} {supervised_error_distribution.mean: 0.3e} [{supervised_error_distribution.std:0.2e}]' \
               f' discriminator {supervised_error_distribution.mean: 0.3e} [{supervised_error_distribution.std:0.2e}]'

    def _train_discriminator_network(self, writer=None) -> str:
        total_error = []
        criterion = nn.BCELoss()
        for sim_batch, real_batch in zip(self.data_loader.sample_shuffled_batch(),
                                         self.discriminator_data_loader.sample_shuffled_batch()):
            self._discriminator_optimizer.zero_grad()
            sim_predictions = torch.cat(self._net.forward_with_all_outputs(sim_batch.observations, train=False))
            real_predictions = torch.cat(self._net.forward_with_all_outputs(real_batch.observations, train=False))
            targets = torch.as_tensor([*[0] * len(sim_predictions), *[1] * len(real_predictions)])\
                .type(self._net.dtype).to(self._device)
            outputs = self._net.discriminate(torch.cat([sim_predictions, real_predictions]), train=True).squeeze(dim=1)
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

    def train(self, epoch: int = -1, writer=None) -> str:
        self.put_model_on_device()

        # Train deep supervision network on training and discriminated unlabeled real data
        message = self._train_main_network(epoch, writer)

        # Train discriminator network
        message += self._train_discriminator_network(writer)

        self.put_model_back_to_original_device()
        return message
