import torch.nn as nn


class BaseNet(nn.Module):

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.input_sizes = [(100, 100, 3)]  # list of tuples for each input argument in the forward pass
        self.dropout_rate = dropout

    def __post_init__(self):
        assert isinstance(self.input_sizes, list) and isinstance(self.input_sizes[0], tuple)
        print('called post init')
