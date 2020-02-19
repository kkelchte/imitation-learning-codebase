import os
import shutil
import subprocess
import shlex
import time
import unittest

import yaml

from src.core.utils import get_filename_without_extension, read_file_to_output, get_file_length
from src.condor.condor_job import CondorJob, CondorJobConfig
from src.condor.helper_functions import create_configs, get_variable_name, strip_variable


class TestCondorJob(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'test_dir/{get_filename_without_extension(__file__)}'

    def test_virtualenv_job(self):
        config_dict = {
            'output_path': self.output_dir,
            'command': 'python3.7 src/condor/test/dummy_python_script.py',
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
        self.assertTrue(len(str(subprocess.check_output(f'condor_q')).split('\\n')) > 10)
        subprocess.call('condor_q')
        while len(str(subprocess.check_output(f'condor_q')).split('\\n')) > 10:
            time.sleep(1)  # Assuming this is only condor job
        for file_path in [job.output_file, job.error_file, job.log_file]:
            self.assertTrue(os.path.isfile(file_path))
        error_file_length = len(open(job.error_file, 'r').readlines())
        self.assertEqual(0, error_file_length)

    def test_singularity_job(self):
        config_dict = {
            'output_path': self.output_dir,
            'command': 'python3.7 src/condor/test/dummy_python_script.py',
            'use_singularity': True
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
        self.assertTrue(len(str(subprocess.check_output(f'condor_q')).split('\\n')) > 10)
        subprocess.call('condor_q')
        while len(str(subprocess.check_output(f'condor_q')).split('\\n')) > 10:
            time.sleep(1)  # Assuming this is only condor job
        for file_path in [job.output_file, job.error_file, job.log_file]:
            self.assertTrue(os.path.isfile(file_path))
        read_file_to_output(os.path.join(self.output_dir, 'condor', condor_dir, 'singularity.output'))
        error_file_length = get_file_length(job.error_file)
        self.assertEqual(error_file_length, 0)

    def test_get_variable_name(self):
        self.assertEqual(get_variable_name('[runner_config][actor_configs][0][file]'), 'file')

    def test_strip_variable(self):
        self.assertEqual(strip_variable('/a/b/c/d.ext'), 'd')
        self.assertEqual(strip_variable(0.001), '1e-03')

    def test_config_creation(self):
        base_file = 'src/scripts/config/data_collection_config.yml'
        with open(base_file, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        config_dict['output_path'] = self.output_dir
        variable_values = ['src/sim/ros/config/actor/ros_expert_noisy.yml',
                           'src/sim/ros/config/actor/ros_expert.yml']
        config_files = create_configs(base_config=config_dict,
                                      adjustments={
                                          '[\"runner_config\"][\"environment_config\"][\"actor_configs\"][0][\"file\"]':
                                              variable_values
                                      })
        self.assertEqual(len(config_files), len(variable_values))
        for index, f in enumerate(config_files):
            self.assertTrue(os.path.isfile(f))
            with open(f, 'r') as fstream:
                config_dict = yaml.load(fstream, Loader=yaml.FullLoader)
                self.assertEqual(config_dict['runner_config']['environment_config']['actor_configs'][0]['file'],
                                 variable_values[index])

    def test_config_creation_with_multiple_variables(self):
        base_file = 'src/scripts/config/evaluate_model_config.yml'
        with open(base_file, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        config_dict['output_path'] = self.output_dir
        actor_tags = ['a', 'b', 'c', 'd']
        actor_config_files = [f'file_{x}' for x in actor_tags]
        adjustments = {
            '[\"data_saver_config\"][\"saving_directory_tag\"]': actor_tags,
            '[\"runner_config\"][\"environment_config\"]'
            '[\"actor_configs\"][0][\"file\"]': actor_config_files
        }
        config_files = create_configs(base_config=config_dict,
                                      adjustments=adjustments)
        self.assertEqual(len(config_files), len(actor_tags))
        for index, f in enumerate(config_files):
            self.assertTrue(os.path.isfile(f))
            with open(f, 'r') as fstream:
                config_dict = yaml.load(fstream, Loader=yaml.FullLoader)
                self.assertEqual(config_dict['runner_config']['environment_config']['actor_configs'][0]['file'],
                                 actor_config_files[index])
                self.assertEqual(config_dict['data_saver_config']['saving_directory_tag'],
                                 actor_tags[index])

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
