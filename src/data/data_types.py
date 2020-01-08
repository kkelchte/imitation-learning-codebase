from typing import List, Tuple

from dataclasses import dataclass
import torch

"""Data types required for converting stored episodes into training batches.

Datatypes required for dataset_saver and dataset_loader.
"""


@dataclass
class InputFrame:
    data: torch.tensor

    def get_size(self) -> Tuple:
        return self.data.size()


@dataclass
class OutputAction:
    data: torch.tensor

    def get_size(self) -> Tuple:
        return self.data.size()


@dataclass
class Batch:
    input: List[InputFrame]
    labels: List[OutputAction]

    def __post_init__(self):
        assert len(self.input) == len(self.labels)