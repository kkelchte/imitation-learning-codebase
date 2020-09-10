import os
import shutil
import time
import unittest
from glob import glob

from src.ai.base_net import ArchitectureConfig
from src.ai.utils import generate_random_dataset_in_raw_data
from src.core.utils import get_to_root_dir, get_filename_without_extension
from src.ai.architectures import *  # Do not remove
from src.scripts.experiment import Experiment, ExperimentConfig


class DatacleaningTest(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

    def test_create_dataset_and_clean(self):
        info = generate_random_dataset_in_raw_data(output_dir=self.output_dir,
                                                   num_runs=20,
                                                   input_size=(100, 100, 3),
                                                   output_size=(1,),
                                                   continuous=True,
                                                   store_hdf5=False)

    def test_clip_first_x_frames(self):
        pass

    def test_split_hdf5_chunks(self):
        pass

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)
        time.sleep(2)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
