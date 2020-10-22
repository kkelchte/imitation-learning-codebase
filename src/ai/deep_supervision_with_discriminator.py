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

    def __init__(self, config: TrainerConfig, network: BaseNet):
        super().__init__(config, network, quiet=True)
        self._logger = get_logger(name=get_filename_without_extension(__file__),
                                  output_path=config.output_path,
                                  quiet=True)
        self._optimizer = eval(f'torch.optim.{self._config.optimizer}')(params=self._net.parameters(),
                                                                        lr=self._config.learning_rate,
                                                                        weight_decay=self._config.weight_decay)
        self._discriminator_optimizer = eval(f'torch.optim.{self._config.optimizer}')(
            params=self._net.discriminator_parameters(),
            lr=self._config.critic_learning_rate if self._config.critic_learning_rate != -1
            else self._config.learning_rate,
            weight_decay=self._config.weight_decay)

        self.discriminator_data_loader = DataLoader(config=self._config.discriminator_data_loader_config)
        self.data_loader.load_dataset()

        lambda_function = lambda f: 1 - f / self._config.scheduler_config.number_of_epochs
        self._scheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer, lr_lambda=lambda_function) \
            if self._config.scheduler_config is not None else None
        cprint(f'Started.', self._logger)

    def train(self, epoch: int = -1, writer=None) -> str:
        # Train deep supervision network
        message = super().train(epoch, writer)
        # Free up RAM before training next
        if self._config.data_loader_config.loop_over_hdf5_files:
            self.data_loader.set_dataset(Dataset())
        # Train deep supervision network

        return message
