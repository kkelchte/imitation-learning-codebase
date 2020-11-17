import os
import shutil
import subprocess
import shlex
import time
import unittest
from glob import glob

import yaml

from src.ai.utils import generate_random_dataset_in_raw_data
from src.core.utils import get_filename_without_extension, read_file_to_output, get_file_length, get_data_dir
from src.condor.condor_job import CondorJob, CondorJobConfig
from src.condor.helper_functions import create_configs, get_variable_name, strip_variable, strip_command, \
    translate_keys_to_string


class TestDagJob(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{get_data_dir(os.environ["PWD"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

    def test_python_job(self):
        config_dict = {
            'output_path': self.output_dir,
            'command': 'python src/condor/test/dummy_python_script.py',
            'use_singularity': False
        }
        config = CondorJobConfig().create(config_dict=config_dict)
        job = CondorJob(config=config)
        condor_dir = sorted(os.listdir(os.path.join(self.output_dir, 'condor')))[-1]
        self.assertTrue(os.path.isdir(os.path.join(self.output_dir, 'condor', condor_dir)))
        job.write_job_file()
        job.write_executable_file()
        for file_path in [job.job_file, job.executable_file]:
            self.assertTrue(os.path.isfile(file_path))
            read_file_to_output(file_path)
        output_executable = subprocess.call(shlex.split(f'{os.path.join(self.output_dir, "condor", condor_dir)}/'
                                                        f'job.executable'))
        self.assertEqual(output_executable, 2)

        self.assertEqual(job.submit(), 0)

        # get job id from log file
        with open(job.log_file, 'r') as f:
            log_line = f.readlines()[0]
            f.close()
        job_id = int(log_line.split(' ')[1].split('.')[0][1:])
        while 'nJobStatus = 2' in str(subprocess.check_output(shlex.split(f'condor_q -l {job_id}'))) or \
                'nJobStatus = 1' in str(subprocess.check_output(shlex.split(f'condor_q -l {job_id}'))):
            print('wait for job to finish...')
            time.sleep(1)

        for file_path in [job.output_file, job.error_file, job.log_file]:
            self.assertTrue(os.path.isfile(file_path))
        error_file_length = len(open(job.error_file, 'r').readlines())
        self.assertEqual(0, error_file_length)

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)
        pass


if __name__ == '__main__':
    unittest.main()
