#!/usr/bin/python3.8
import torch
from torch import nn
import numpy as np

from src.ai.base_net import BaseNet
from src.ai.domain_adaptation_trainer import DomainAdaptationTrainer
from src.ai.trainer import TrainerConfig, Trainer
from src.ai.losses import *
from src.ai.deep_supervision import DeepSupervision
from src.ai.utils import get_reward_to_go, get_checksum_network_parameters, data_to_tensor, plot_gradient_flow
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
        task_error = []
        domain_error = []
        for source_batch, target_batch in zip(self.data_loader.sample_shuffled_batch(),
                                              self.target_data_loader.sample_shuffled_batch()):
            self._optimizer.zero_grad()
            targets = data_to_tensor(source_batch.actions).type(self._net.dtype).to(self._device)

            # deep supervision loss
            probabilities = self._net.forward_with_all_outputs(source_batch.observations, train=True)
            task_loss = self._criterion(probabilities[-1], targets).mean()
            for index, prob in enumerate(probabilities[:-1]):
                task_loss += self._criterion(prob, targets).mean()
            task_loss *= (1 - self._config.epsilon)

            # add domain adaptation loss on distribution of output pixels at each output
            domain_loss = sum(self._domain_adaptation_criterion(sp, tp) for sp, tp in zip(
                self._net.get_features(source_batch.observations, train=True),
                self._net.get_features(target_batch.observations, train=True)
            )) * self._config.epsilon

            loss = task_loss + domain_loss
            loss.backward()
            if self._config.gradient_clip_norm != -1:
                nn.utils.clip_grad_norm_(self._net.parameters(),
                                         self._config.gradient_clip_norm)
            self._optimizer.step()
            self._net.global_step += 1
            task_error.append(task_loss.cpu().detach().numpy())
            domain_error.append(domain_loss.cpu().detach().numpy())
            total_error.append(loss.cpu().detach().numpy())

        self.put_model_back_to_original_device()

        if self._scheduler is not None:
            self._scheduler.step()

        task_error_distribution = Distribution(task_error)
        domain_error_distribution = Distribution(domain_error)
        total_error_distribution = Distribution(total_error)
        if writer is not None:
            writer.set_step(self._net.global_step)
            writer.write_distribution(task_error_distribution, 'training/task_error')
            writer.write_distribution(domain_error_distribution, 'training/domain_error')
            writer.write_distribution(total_error_distribution, 'training/total_error')
            if self._config.store_output_on_tensorboard and epoch % 30 == 0:
                writer.write_output_image(probabilities[-1], 'source/predictions')
                writer.write_output_image(targets, 'source/targets')
                writer.write_output_image(torch.stack(source_batch.observations), 'source/inputs')
                writer.write_output_image(self._net.forward(target_batch.observations, train=False),
                                          'target/predictions')
                writer.write_output_image(torch.stack(target_batch.observations), 'target/inputs')
            if self._config.store_feature_maps_on_tensorboard and epoch % 30 == 0:
                for name, batch in zip(['source', 'target'], [source_batch, target_batch]):
                    outputs = self._net.forward_with_intermediate_outputs(batch.observations, train=False)
                    for i in range(4):  # store first 5 images of batch
                        for layer in ['x1', 'x2', 'x3', 'x4']:
                            feature_maps = outputs[layer][i].flatten(start_dim=0, end_dim=0)
                            title = f'feature_map/{name}/layer_{layer}/{i}'
                            # title += 'inds_' + '_'.join([str(v.item()) for v in winning_indices.indices])
                            # title += '_vals_' + '_'.join([f'{v.item():0.2f}' for v in winning_indices.values])
                            writer.write_output_image(feature_maps, title)
            writer.write_figure(plot_gradient_flow(self._net.named_parameters()))

        return f' task {self._config.criterion} ' \
               f'{task_error_distribution.mean: 0.3e} ' \
               f'[{task_error_distribution.std:0.2e}] ' \
               f' domain {self._config.domain_adaptation_criterion} ' \
               f'{domain_error_distribution.mean: 0.3e} ' \
               f'[{domain_error_distribution.std: 0.2e}]'
