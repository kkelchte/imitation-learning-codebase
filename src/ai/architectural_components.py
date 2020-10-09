from typing import Tuple, Type

import torch


class ResidualBlock(torch.nn.Module):

    def __init__(self, input_channels,
                 output_channels,
                 batch_norm: bool = True,
                 activation: torch.nn.ReLU = torch.nn.ReLU(),
                 pool: torch.nn.MaxPool2d = None,
                 strides: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (0, 0),
                 kernel_sizes: Tuple[int, int] = (3, 3)):
        """
        Create a residual block
        :param input_channels: number of input channels at input
        :param output_channels: number of input channels at input
        :param batch_norm: bool specifying to use batch norm 2d (True)
        :param activation: specify torch nn module activation (ReLU)
        :param pool: specify pooling layer applied as first layer
        :param strides: tuple specifying the stride and so the down sampling
        """
        super().__init__()
        self._down_sample = torch.nn.Conv2d(input_channels, output_channels, kernel_size=1,
                                            stride=sum([s - 1 for s in strides]) + (1 if pool is None else 2)) \
            if sum([s - 1 for s in strides]) > 0 or pool is not None else torch.nn.Identity()
        self._final_activation = activation
        elements = []
        if pool is not None:
            elements.append(pool)
        elements.append(torch.nn.Conv2d(in_channels=input_channels,
                                        out_channels=output_channels,
                                        kernel_size=kernel_sizes[0],
                                        padding=padding[0],
                                        stride=strides[0]))
        if batch_norm:
            elements.append(torch.nn.BatchNorm2d(output_channels))
        elements.append(activation)
        elements.append(torch.nn.Conv2d(in_channels=output_channels,
                                        out_channels=output_channels,
                                        kernel_size=kernel_sizes[1],
                                        padding=padding[1],
                                        stride=strides[1]))
        if batch_norm:
            elements.append(torch.nn.BatchNorm2d(output_channels))
        elements.append(activation)
        self.residual_net = torch.nn.Sequential(*elements)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.residual_net(inputs)
        x += self._down_sample(inputs)
        return self._final_activation(x)
