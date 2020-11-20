#!/bin/python3.8
import torch

from src.ai.base_net import ArchitectureConfig
from src.ai.architectures.bc_deeply_supervised_auto_encoder import Net as BaseNet
from src.ai.architectures.bc_deeply_supervised_auto_encoder import ImageNet
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension

"""
Four encoding and four decoding layers with sharing weights over different res blocks
Expects 3x200x200 inputs and outputs 200x200
Output layer is modified as heuristic filter which selects most promising feature maps based on distribution similarity.
"""


class Net(BaseNet):

    def __init__(self, config: ArchitectureConfig, quiet: bool = False):
        super().__init__(config=config, quiet=True)
        self.label_mean = 0.0125
        self.label_std = 0.1075
        if not quiet:
            self._logger = get_logger(name=get_filename_without_extension(__file__),
                                      output_path=config.output_path,
                                      quiet=False)

            self.initialize_architecture()
            cprint(f'Started.', self._logger)

    def select_and_combine_best_feature_maps(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Combine feature maps as weighted sum where weights corresponds to similarity to target pixel distribution.
        Weights are normalized so they sum to one and are estimated without gradients.
        inputs: (N, C, H, W) tensor with C feature maps
        returns: (N, 1, H, W)
        """
        means = torch.flatten(inputs.detach(), start_dim=2, end_dim=3).mean(dim=2)  # N x 32
        # stds = torch.flatten(inputs.detach(), start_dim=2, end_dim=3).std(dim=2)  # N x 32
        # weights = 1 / ((means - self.label_mean)**2 + (stds - self.label_std)**2)  # N x 32
        weights = 1 / ((means - self.label_mean)**2)  # N x 32
        normalized_weights = weights / weights.sum(dim=1, keepdims=True)  # N x 32
        return torch.mul(normalized_weights.unsqueeze(2).unsqueeze(2).repeat(1, 1,
                                                                             inputs.shape[2], inputs.shape[3]).detach(),
                         inputs).sum(dim=1).unsqueeze(dim=1)

    def forward_with_intermediate_outputs(self, inputs, train: bool = False) -> dict:
        self.set_mode(train)
        processed_inputs = self.process_inputs(inputs)

        results = {'x1': self.residual_1(self.conv0(processed_inputs))}
        results['out1'] = self.select_and_combine_best_feature_maps(results['x1'])
        # results['out1'] = self.side_logit_1(results['x1'])
        results['prob1'] = self.sigmoid(results['out1']).squeeze(dim=1)

        results['x2'] = self.residual_2(results['x1'])
        results['out2'] = self.select_and_combine_best_feature_maps(results['x2'])
        # results['out2'] = self.side_logit_2(results['x2'])
        results['prob2'] = self.upsample_2(self.sigmoid(results['out2'])).squeeze(dim=1)

        results['x3'] = self.residual_3(results['x2'])
        results['out3'] = self.select_and_combine_best_feature_maps(results['x3'])
        # results['out3'] = self.side_logit_3(results['x3'])
        results['prob3'] = self.upsample_3(self.sigmoid(results['out3'])).squeeze(dim=1)

        results['x4'] = self.residual_4(results['x3'])
        results['out4'] = self.select_and_combine_best_feature_maps(results['x4'])
        # results['out4'] = self.side_logit_4(results['x4'])
        results['prob4'] = self.upsample_4(self.sigmoid(results['out4'])).squeeze(dim=1)

        final_logit = self.weight_1 * results['out1'] + \
                      self.weight_2 * self.upsample_2(results['out2']) + \
                      self.weight_3 * self.upsample_3(results['out3']) + \
                      self.weight_4 * self.upsample_4(results['out4'])
        results['final_prob'] = self.sigmoid(final_logit).squeeze(dim=1)
        return results
