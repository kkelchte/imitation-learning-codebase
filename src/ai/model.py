#!/usr/bin/python3.7
import os
from dataclasses import dataclass
from typing import List

import torch
from dataclasses_json import dataclass_json

from src.ai.architectures import *  # Do not remove
from src.core.config_loader import Config
from src.core.logger import get_logger, cprint
"""
Model contains of architectures (can be modular).
Model ensures proper initialization and storage of model parameters.
"""


@dataclass_json
@dataclass
class ModelConfig(Config):
    load_checkpoint_path: str = None
    architecture: str = None
    dropout: float = 0.

    def __post_init__(self):
        if self.load_checkpoint_path is None:
            del self.load_checkpoint_path


class Model:

    def __init__(self, config: ModelConfig):
        self._config = config
        self._logger = get_logger(name=__name__,
                                  output_path=config.output_path,
                                  quite=False)
        self._checkpoint_directory = os.path.join(self._config.output_path, 'torch_checkpoints')
        cprint(f'Started.', self._logger)
        self._architecture = eval(f'{self._config.architecture}.Net('
                                  f'dropout={self._config.dropout})')
        self.load_from_checkpoint()

    def forward(self, inputs: List[torch.Tensor], train: bool = False):
        return self._architecture.forward(*inputs, train=train)

    def load_from_checkpoint(self):
        # Priority one: latest checkpoint in output_path/
        pass
        # model = ConfidenceNet(model_config)
        # if os.path.exists(model_path):
        #     try:
        #         model.load_state_dict(torch.load(model_path))
        #         print(f'{os.path.basename(__file__)}: loaded weights from {model_path} with confidence.')
        #     except RuntimeError:
        #         model = Net(model_config)
        #         model.load_state_dict(torch.load(model_path))
        #         print(f'{os.path.basename(__file__)}: loaded weights from {model_path} without confidence.')
        # return model

    def save_to_checkpoint(self, tag: str = ''):
        filename = f'checkpoint_{tag}' if tag != '' else 'checkpoint'
        torch.save(self._architecture.state_dict(), f'{self._checkpoint_directory}/{filename}')
        torch.save(self._architecture.state_dict(), f'{self._checkpoint_directory}/checkpoint_latest')

    def get_input_sizes(self):
        return self._architecture.input_sizes

    def get_output_sizes(self):
        return self._architecture.output_sizes

    def get_parameters(self):
        return self._architecture.parameters()
