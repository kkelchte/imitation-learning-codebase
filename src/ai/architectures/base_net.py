from typing import Tuple, List, Any, Union
from typing_extensions import Protocol, runtime_checkable

import torch.nn as nn


@runtime_checkable
class GetItem(Protocol):
    def __getitem__(self: 'GetItem', key: Any) -> Any: pass


def assign(value: Union[GetItem, int]) -> List[Tuple]:
    if isinstance(value, GetItem):
        if isinstance(value[0], GetItem):
            return [tuple(v) for v in value]
        else:
            return [tuple(value)]
    else:
        return [tuple((value,))]


class BaseNet(nn.Module):

    def __init__(self, input_sizes: Union[GetItem, int], output_sizes: Union[GetItem, int]):
        super().__init__()
        self.input_sizes = []
        self.output_sizes = []
        self.input_sizes = assign(input_sizes)
        self.output_sizes = assign(output_sizes)


