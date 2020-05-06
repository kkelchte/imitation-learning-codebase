
from torch.utils.tensorboard import SummaryWriter

from src.core.data_types import Distribution


class TensorboardWrapper(SummaryWriter):

    def __init__(self, log_dir: str):
        super().__init__(log_dir)
        self.step = 0

    def set_step(self, step: int):
        self.step = step

    def increment_step(self):
        self.step += 1

    def write_distribution(self, distribution: Distribution, tag: str = ""):
        self.add_scalar(f"{tag}_mean" if tag != "" else "mean", distribution.mean, global_step=self.step)
        self.add_scalar(f"{tag}_std" if tag != "" else "std", distribution.std, global_step=self.step)

    def write_scalar(self, data: float, tag: str):
        self.add_scalar(tag, data, global_step=self.step)
