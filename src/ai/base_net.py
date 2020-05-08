import os
import time

import torch
import numpy as np
from dataclasses import dataclass
from typing import Union, Any, Optional

from dataclasses_json import dataclass_json

import torch.nn as nn
from typing_extensions import runtime_checkable, Protocol

from src.ai.utils import get_checksum_network_parameters
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
    load_checkpoint_dir: Optional[str] = None  # path to checkpoints
    initialisation_type: str = 'xavier'
    initialisation_seed: int = 0
    device: str = 'cpu'
    weight_decay: Union[float, str] = 'default'
    dropout: Union[float, str] = 'default'
    dtype: str = 'default'
    log_std: Union[float, str] = 'default'

    def __post_init__(self):
        if self.load_checkpoint_dir is None:
            del self.load_checkpoint_dir


class BaseNet(nn.Module):

    def __init__(self, config: ArchitectureConfig, quiet: bool = True):
        super().__init__()
        self.input_size = None
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

    def load_network_weights(self):
        if self._config.load_checkpoint_dir:
            self._config.load_checkpoint_dir = self._config.load_checkpoint_dir \
                if self._config.load_checkpoint_dir.endswith('torch_checkpoints') \
                else os.path.join(self._config.load_checkpoint_dir, 'torch_checkpoints')
            self.load_from_checkpoint(checkpoint_dir=self._config.load_checkpoint_dir)
        else:
            self.initialize_architecture_weights(self._config.initialisation_type)
        cprint(f"network checksum: {get_checksum_network_parameters(self.parameters())}", self._logger)

    def initialize_architecture_weights(self, initialisation_type: str = 'xavier'):
        torch.manual_seed(self._config.initialisation_seed)
        for p in self.parameters():
            if initialisation_type == 'xavier':
                if len(p.shape) == 1:
                    nn.init.uniform_(p, a=-0.03, b=0.03)
                else:
                    nn.init.xavier_uniform_(p)
            elif initialisation_type == 'constant':
                nn.init.constant_(p, 0.03)
            else:
                raise NotImplementedError

    def load_from_checkpoint(self, checkpoint_dir: str):
        if len([f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]) == 0:
            cprint(f'Could not find suitable checkpoint in {checkpoint_dir}', self._logger, MessageType.error)
            time.sleep(0.5)
            raise FileNotFoundError
        # Get latest checkpoint file
        if os.path.isfile(os.path.join(checkpoint_dir, 'checkpoint_best.ckpt')):
            checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint_best.ckpt')
        elif os.path.isfile(os.path.join(checkpoint_dir, 'checkpoint_latest.ckpt')):
            checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint_latest.ckpt')
        else:
            checkpoints = {int(f.split('.')[0].split('_')[-1]):
                           os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir)}
            checkpoint_file = checkpoints[max(checkpoints.keys())]
        # Load params internally and return checkpoint
        checkpoint = torch.load(checkpoint_file)
        self.global_step = checkpoint['global_step']
        del checkpoint['global_step']
        self.load_state_dict(checkpoint['model_state'])
        del checkpoint['model_state']
        self.extra_checkpoint_info = checkpoint  # store extra parameters e.g. optimizer / loss params

    def save_to_checkpoint(self, tag: str = '', extra_info: dict = None):
        filename = f'checkpoint_{tag}' if tag != '' else 'checkpoint'
        filename += '.ckpt'
        checkpoint = {
            'global_step': self.global_step,
            'model_state': self.state_dict()
        }
        if extra_info is not None:
            for k in extra_info.keys():
                checkpoint[k] = extra_info[k]
        torch.save(checkpoint, f'{self._checkpoint_output_directory}/{filename}')
        torch.save(checkpoint, f'{self._checkpoint_output_directory}/checkpoint_latest.ckpt')
        cprint(f'stored {filename}', self._logger)

    def set_device(self, device: str):
        self._device = torch.device(
            "cuda" if device in ['gpu', 'cuda'] and torch.cuda.is_available() else "cpu"
        )
        self.to(self._device)

    def forward(self, inputs: Union[torch.Tensor, np.ndarray, list, int, float], train: bool) -> torch.Tensor:
        # adjust gradient saving
        if train:
            self.train()
        else:
            self.eval()
        # preprocess inputs
        if not isinstance(inputs, torch.Tensor):
            try:
                inputs = torch.as_tensor(inputs, dtype=self.dtype)
            except ValueError:
                inputs = torch.stack(inputs).type(self.dtype)
        inputs = inputs.type(self.dtype)
        # swap H, W, C --> C, H, W
        if torch.argmin(torch.as_tensor(inputs.size())) != 0 \
                and self.input_size[0] == inputs.size()[-1]\
                and len(self.input_size) == len(inputs.size()):
            inputs = inputs.permute(2, 0, 1)

        # swap B, H, W, C --> B, C, H, W
        if len(self.input_size) + 1 == len(inputs.size()) and \
                torch.argmin(torch.as_tensor(inputs.size()[1:])) != 0 \
                and self.input_size[0] == inputs.size()[-1]:
            inputs = inputs.permute(0, 3, 1, 2)

        # add batch dimension if required
        if len(self.input_size) == len(inputs.size()):
            inputs = inputs.unsqueeze(0)

        # put inputs on device
        inputs.to(self._device)

        return inputs

    def get_action(self, inputs, train: bool = False) -> Action:
        raise NotImplementedError

    def get_device(self) -> torch.device:
        return self._device

    def to_device(self, device: torch.device) -> None:
        self.to(device)
        self._device = device

    def count_parameters(self) -> int:
        count = 0
        for p in self.parameters():
            count += np.prod(p.shape)
        return count

    def remove(self):
        [h.close() for h in self._logger.handlers]
