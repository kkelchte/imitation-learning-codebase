import os
import time

import torch
import numpy as np
from cv2 import cv2
from dataclasses import dataclass
from typing import Union, Any, Optional

from dataclasses_json import dataclass_json

import torch.nn as nn
from typing_extensions import runtime_checkable, Protocol

from src.ai.utils import get_checksum_network_parameters, initialize_weights
from src.core.config_loader import Config
from src.core.data_types import Action
from src.core.logger import get_logger, cprint, MessageType
from src.core.utils import get_filename_without_extension

"""
BaseClass for neural network architectures.
Include functionality to initialize network, load and store checkpoints according to config.
All other architectures in src.ai.architectures should inherit from this class.
"""


@runtime_checkable
class GetItem(Protocol):
    def __getitem__(self: 'GetItem', key: Any) -> Any: pass


@dataclass_json
@dataclass
class ArchitectureConfig(Config):
    architecture: str = None  # name of architecture to be loaded
    initialisation_type: str = 'xavier'
    random_seed: int = 0
    device: str = 'cpu'
    finetune: bool = False
    weight_decay: Union[float, str] = 'default'
    dropout: Union[float, str] = 'default'
    batch_normalisation: Union[bool, str] = 'default'
    dtype: str = 'default'
    log_std: Union[float, str] = 'default'


class BaseNet(nn.Module):

    def __init__(self, config: ArchitectureConfig, quiet: bool = True):
        super().__init__()
        self.input_size = None
        # input range from 0 -> 1 is expected, for range -1 -> 1 this field should state 'zero_centered'
        self.input_scope = 'default'
        self.output_size = None
        self.discrete = None
        self._config = config
        self.dtype = torch.float32 if config.dtype == 'default' else eval(f"torch.{config.dtype}")

        if not quiet:
            self._logger = get_logger(name=get_filename_without_extension(__file__),
                                      output_path=config.output_path,
                                      quiet=True)
            cprint(f'Started.', self._logger)
        self._checkpoint_output_directory = os.path.join(self._config.output_path, 'torch_checkpoints')
        os.makedirs(self._checkpoint_output_directory, exist_ok=True)

        self.extra_checkpoint_info = None
        self._device = torch.device(
            "cuda" if self._config.device in ['gpu', 'cuda'] and torch.cuda.is_available() else "cpu"
        )

        self.global_step = torch.as_tensor(0, dtype=torch.int32)

    def initialize_architecture(self):
        torch.manual_seed(self._config.random_seed)
        for layer in self.modules():
            initialize_weights(layer, initialisation_type=self._config.initialisation_type)

    def get_checksum(self):
        return get_checksum_network_parameters(self.parameters())

    def set_device(self, device: Union[str, torch.device]):
        self._device = torch.device(
            "cuda" if device in ['gpu', 'cuda'] and torch.cuda.is_available() else "cpu"
        ) if isinstance(device, str) else device
        try:
            self.to(self._device)
        except AssertionError:
            cprint(f'failed to work on {self._device} so working on cpu', self._logger, msg_type=MessageType.warning)
            self._device = torch.device('cpu')
            self.to(self._device)

    def forward(self, inputs: Union[torch.Tensor, np.ndarray, list, int, float], train: bool) -> torch.Tensor:
        # adjust gradient saving
        if train:
            self.train()
        else:
            self.eval()
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        if len(self.input_size) == 3:
            # check if 2D input is correct
            # compare channel first / last for single image:
            if len(inputs.shape) == 3 and inputs.shape[-1] == self.input_size[0]:
                # in case it's channel last, assume single raw data input which requires preprocess:
                # check for size
                if inputs.shape[1] != self.input_size[1]:
                    # resize with opencv
                    inputs = cv2.resize(np.asarray(inputs), dsize=(self.input_size[1], self.input_size[2]),
                                        interpolation=cv2.INTER_LANCZOS4)
                    if self.input_size[0] == 1:
                        inputs = inputs.mean(axis=-1, keepdims=True)
                # check for scope
                if inputs.max() > 1 or inputs.min() < 0:
                    inputs += inputs.min()
                    inputs /= inputs.max()
                if self.input_scope == 'zero_centered':
                    inputs *= 2
                    inputs -= 1
                # make channel first and add batch dimension
                inputs = torch.as_tensor(inputs).permute(2, 0, 1).unsqueeze(0)

        # create Tensors
        if not isinstance(inputs, torch.Tensor):
            try:
                inputs = torch.as_tensor(inputs, dtype=self.dtype)
            except ValueError:
                inputs = torch.stack(inputs).type(self.dtype)
        inputs = inputs.type(self.dtype)

        # add batch dimension if required
        if len(self.input_size) == len(inputs.size()):
            inputs = inputs.unsqueeze(0)

        # put inputs on device
        inputs = inputs.to(self._device)

        return inputs

    def get_action(self, inputs) -> Action:
        raise NotImplementedError

    def get_device(self) -> torch.device:
        return self._device

    def count_parameters(self) -> int:
        count = 0
        for p in self.parameters():
            count += np.prod(p.shape)
        return count

    def remove(self):
        [h.close() for h in self._logger.handlers]

    def get_checkpoint(self) -> dict:
        """
        :return: a dictionary with global_step and model_state of neural network.
        """
        return {
            'global_step': self.global_step,
            'model_state': self.state_dict()
        }

    def load_checkpoint(self, checkpoint) -> None:
        """
        Try to load checkpoint in global step and model state. Raise error.
        :param checkpoint: dictionary containing 'global step' and 'model state'
        :return: None
        """
        self.global_step = checkpoint['global_step']
        self.load_state_dict(checkpoint['model_state'])
        self.set_device(self._device)
