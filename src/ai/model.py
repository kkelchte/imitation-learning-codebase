#!/usr/bin/python3.7
import os
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple, Union

import torch
from torch import nn
import numpy as np
from dataclasses_json import dataclass_json

from src.ai.architectures import *  # Do not remove
from src.core.config_loader import Config
from src.core.logger import get_logger, cprint, MessageType
from src.core.utils import get_filename_without_extension

"""
Model contains an open architecture field used by evaluator, trainer and experiment.
Model serves as a wrapper over the architecture to:
    - load and store checkpoint: [can be added to architecture in the end]
    - initialize architecture
    
"""





@dataclass_json
@dataclass
class ModelConfig(Config):
    load_checkpoint_dir: str = None
    architecture: str = None
    dropout: float = 0.
    input_sizes: Union[Tuple, List, int] = None
    output_sizes: Union[Tuple, List, int] = None
    initialisation_type: InitializationType = InitializationType.Xavier
    pretrained: bool = False
    initialisation_seed: int = 0
    device: str = 'cpu'
    discrete: bool = False

    def __post_init__(self):
        if self.load_checkpoint_dir is None:
            del self.load_checkpoint_dir


class Model:

    def __init__(self, config: ModelConfig):
        self._config = config
        self.discrete = self._config.discrete
        self._logger = get_logger(name=get_filename_without_extension(__file__),
                                  output_path=config.output_path,
                                  quite=False)
        cprint(f'Started.', self._logger)
        self._checkpoint_directory = os.path.join(self._config.output_path, 'torch_checkpoints')
        os.makedirs(self._checkpoint_directory, exist_ok=True)
        self._architecture = eval(self._config.architecture).Net(
                                input_sizes=self._config.input_sizes,
                                output_sizes=self._config.output_sizes,
                                dropout=self._config.dropout,)

        if self._config.load_checkpoint_dir:
            self._config.load_checkpoint_dir = self._config.load_checkpoint_dir \
                if self._config.load_checkpoint_dir.endswith('torch_checkpoints') \
                else os.path.join(self._config.load_checkpoint_dir, 'torch_checkpoints')
            self.load_from_checkpoint(checkpoint_dir=self._config.load_checkpoint_dir)
        else:
            self.initialize_architecture_weights(self._config.initialisation_type)

        self._device = torch.device(
            "cuda" if self._config.device in ['gpu', 'cuda'] and torch.cuda.is_available() else "cpu"
        )



    def forward(self, inputs: List[torch.Tensor], train: bool = False):
        return self._architecture.forward(inputs=self.process_input(inputs), train=train)

    def get_input_sizes(self):
        return self._architecture.input_sizes

    def get_output_sizes(self):
        return self._architecture.output_sizes

    def get_parameters(self) -> list:
        return list(self._architecture.parameters())

    def set_device(self, device: str):
        self._device = torch.device(
            "cuda" if device in ['gpu', 'cuda'] and torch.cuda.is_available() else "cpu"
        )
        self._architecture.to(self._device)

    def get_device(self):
        return self._device.type
