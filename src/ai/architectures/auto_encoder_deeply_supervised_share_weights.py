#!/bin/python3.8

from src.ai.base_net import ArchitectureConfig
from src.ai.architectures.bc_deeply_supervised_auto_encoder import Net as BaseNet
from src.ai.architectures.bc_deeply_supervised_auto_encoder import ImageNet
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension

"""
Four encoding and four decoding layers with sharing weights over different res blocks
Expects 3x200x200 inputs and outputs 200x200
"""


class Net(BaseNet):

    def __init__(self, config: ArchitectureConfig, quiet: bool = False):
        super().__init__(config=config, quiet=True)

        self.residual_2.residual_net[0].weight = self.residual_1.residual_net[0].weight
        self.residual_2.residual_net[0].bias = self.residual_1.residual_net[0].bias
        self.residual_2.residual_net[2].weight = self.residual_1.residual_net[2].weight
        self.residual_2.residual_net[2].bias = self.residual_1.residual_net[2].bias

        self.residual_3.residual_net[0].weight = self.residual_1.residual_net[0].weight
        self.residual_3.residual_net[0].bias = self.residual_1.residual_net[0].bias
        self.residual_3.residual_net[2].weight = self.residual_1.residual_net[2].weight
        self.residual_3.residual_net[2].bias = self.residual_1.residual_net[2].bias

        self.residual_4.residual_net[0].weight = self.residual_1.residual_net[0].weight
        self.residual_4.residual_net[0].bias = self.residual_1.residual_net[0].bias
        self.residual_4.residual_net[2].weight = self.residual_1.residual_net[2].weight
        self.residual_4.residual_net[2].bias = self.residual_1.residual_net[2].bias

        if not quiet:
            self._logger = get_logger(name=get_filename_without_extension(__file__),
                                      output_path=config.output_path,
                                      quiet=False)

            self.initialize_architecture()
            cprint(f'Started.', self._logger)
