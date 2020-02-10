#!/usr/bin/python3.7
from dataclasses import dataclass
from typing import List

import torch
from dataclasses_json import dataclass_json

from src.ai.architectures import *  # Do not remove
from src.core.config_loader import Config
from src.core.logger import get_logger, cprint
"""
Model contains of architectures (can be modular) and components (such as losses).

"""


@dataclass_json
@dataclass
class ModelConfig(Config):
    architecture: str = None
    dropout: float = 0.


class Model:

    def __init__(self, config: ModelConfig):
        self._config = config
        self._logger = get_logger(name=__name__,
                                  output_path=config.output_path,
                                  quite=False)
        cprint(f'Started.', self._logger)
        self._architecture = eval(f'{self._config.architecture}.Net('
                                  f'dropout={self._config.dropout})')

    def forward(self, inputs: List[torch.Tensor]):
        return self._architecture.forward(*inputs)

    def load_from_checkpoint(self):
        pass

    def save_to_checkpoint(self):
        pass

    def get_input_sizes(self):
        return self._architecture.input_sizes
