import os
import shutil
import subprocess
import shlex
import time
import unittest
from glob import glob

import yaml

from src.ai.utils import generate_random_dataset_in_raw_data
from src.condor.test.test_condor_job import wait_for_job_to_finish
from src.core.utils import get_filename_without_extension, read_file_to_output, get_file_length, get_data_dir
from src.condor.condor_job import CondorJob, CondorJobConfig
from src.condor.helper_functions import create_configs, get_variable_name, strip_variable, strip_command, \
    translate_keys_to_string


def create_condor_job(output_path) -> CondorJob:
    config_dict = {
        'output_path': output_path,
        'command': 'python src/condor/test/dummy_python_script.py',
    }
    config = CondorJobConfig().create(config_dict=config_dict)
    job = CondorJob(config=config)
    job.write_job_file()
    job.write_executable_file()
    return job


class TestDagJob(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{get_data_dir(os.environ["PWD"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

    def test_python_job(self):
        jobs = []
        for job_index in range(2):
            jobs.append(create_condor_job(os.path.join(self.output_dir, job_index)))

        wait_for_job_to_finish(job.log_file)

        for file_path in [job.output_file, job.error_file, job.log_file]:
            self.assertTrue(os.path.isfile(file_path))
        with open(job.error_file, 'r') as f:
            error_file_length = len(f.readlines())
        self.assertEqual(0, error_file_length)

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)
        pass


if __name__ == '__main__':
    unittest.main()
