#!/bin/python3.8
from typing import Tuple

import torch
import torch.nn as nn

from src.ai.base_net import ArchitectureConfig
from src.ai.architectures.bc_deeply_supervised_auto_encoder import Net as BaseNet
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension

"""
Four encoding and four decoding layers with dropout.
Expects 3x200x200 inputs and outputs 200x200
"""


class Net(BaseNet):

    def __init__(self, config: ArchitectureConfig, quiet: bool = False):
        super().__init__(config=config, quiet=True)

        self.side_conf_1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.side_conf_2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.side_conf_3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.side_conf_4 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)

        if not quiet:
            self._logger = get_logger(name=get_filename_without_extension(__file__),
                                      output_path=config.output_path,
                                      quiet=False)
            self.initialize_architecture()
            cprint(f'Started.', self._logger)

    def forward_with_all_outputs(self, inputs, train: bool = False) -> Tuple[Tuple[torch.Tensor,
                                                                                   torch.Tensor,
                                                                                   torch.Tensor,
                                                                                   torch.Tensor],
                                                                             Tuple[torch.Tensor,
                                                                                   torch.Tensor,
                                                                                   torch.Tensor,
                                                                                   torch.Tensor], torch.Tensor]:
        results = self.forward_with_intermediate_outputs(inputs, train)
        conf1 = self.sigmoid(self.side_conf_1(results['x1'])).squeeze(dim=1)
        conf2 = self.upsample_2(self.sigmoid(self.side_conf_2(results['x2']))).squeeze(dim=1)
        conf3 = self.upsample_3(self.sigmoid(self.side_conf_3(results['x3']))).squeeze(dim=1)
        conf4 = self.upsample_4(self.sigmoid(self.side_conf_4(results['x4']))).squeeze(dim=1)
        return (results['prob1'], results['prob2'], results['prob3'], results['prob4']), \
               (conf1, conf2, conf3, conf4), results['final_prob']
