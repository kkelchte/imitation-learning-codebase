import os
import shutil
import subprocess
import shlex
import time
import unittest
from glob import glob

import yaml

from src.core.utils import get_filename_without_extension, read_file_to_output, get_file_length
from src.condor.condor_job import CondorJob, CondorJobConfig
from src.condor.helper_functions import create_configs, get_variable_name, strip_variable, strip_command


class TestCondorJob(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

    @unittest.skip
    def test_virtualenv_job(self):
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
        self.assertTrue(len(str(subprocess.check_output(f'condor_q')).split('\\n')) > 10)
        subprocess.call('condor_q')
        while len(str(subprocess.check_output(f'condor_q')).split('\\n')) > 10:
            time.sleep(1)  # Assuming this is only condor job
        for file_path in [job.output_file, job.error_file, job.log_file]:
            self.assertTrue(os.path.isfile(file_path))
        error_file_length = len(open(job.error_file, 'r').readlines())
        self.assertEqual(0, error_file_length)

    @unittest.skip
    def test_singularity_job(self):
        config_dict = {
            'output_path': self.output_dir,
            'command': 'python src/condor/test/dummy_python_script.py',
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

    def test_strip_command(self):
        result = strip_command('python src/scripts/dataset_experiment.py --config src/scripts/config/train.yml')
        self.assertEqual(result, 'dataset_experiment')

    def test_config_creation(self):
        config_dict = {'output_path': 'experimental_data/cube_world',
                       'number_of_epochs': 1,
                       'number_of_episodes': 3,
                       'environment_config': {'factory_key': 'ROS',
                                              'max_number_of_steps': 1000,
                                              'ros_config': {'info': '-current_waypoint -sensor/odometry',
                                                             'observation': 'forward_camera',
                                                             'visible_xterm': False, 'step_rate_fps': 30,
                                                             'ros_launch_config': {'random_seed': 123,
                                                                                   'robot_name': 'turtlebot_sim',
                                                                                   'fsm_config': 'single_run',
                                                                                   'fsm': True,
                                                                                   'control_mapping': True,
                                                                                   'waypoint_indicator': True,
                                                                                   'control_mapping_config': 'default',
                                                                                   'world_name': 'cube_world',
                                                                                   'x_pos': 0.0, 'y_pos': 0.0,
                                                                                   'z_pos': 0.0, 'yaw_or': 1.57,
                                                                                   'gazebo': True},
                                                             'actor_configs': [{'name': 'ros_expert',
                                                                                'file': 'wrong_path.yml'}]}},
                       'data_saver_config': {'training_validation_split': 0.9,
                                             'store_hdf5': True,
                                             'separate_raw_data_runs': True,
                                             'saving_directory_tag': 'expert'}}

        variable_values = ['src/sim/ros/config/actor/ros_expert_noisy.yml',
                           'src/sim/ros/config/actor/ros_expert.yml']
        config_files = create_configs(base_config=config_dict,
                                      output_path=self.output_dir,
                                      adjustments={
                                          '[\"environment_config\"][\"ros_config\"][\"actor_configs\"][0][\"file\"]':
                                              variable_values
                                      })
        self.assertEqual(len(config_files), len(variable_values))
        for index, f in enumerate(config_files):
            self.assertTrue(os.path.isfile(f))
            with open(f, 'r') as fstream:
                config_dict = yaml.load(fstream, Loader=yaml.FullLoader)
                self.assertEqual(config_dict['environment_config']['ros_config']['actor_configs'][0]['file'],
                                 variable_values[index])

    @unittest.skip
    def test_local_storage_with_nested_directories(self):
        # create some already existing nested directory:
        nested_path = os.path.join(self.output_dir, 'nested_dir_1', 'nested_dir_2')
        os.makedirs(nested_path, exist_ok=True)
        pre_existing_file = os.path.join(nested_path, 'already_existing_file')
        with open(pre_existing_file, 'w') as f:
            f.write('pre_existed')
        # launch job
        config_dict = {
            'output_path': self.output_dir,
            'command': 'python src/condor/test/dummy_python_script.py --config src/condor/test/dummy_config.yml',
            'use_singularity': True,
            'save_locally': True
        }
        config = CondorJobConfig().create(config_dict=config_dict)
        job = CondorJob(config=config)
        condor_dir = sorted(os.listdir(os.path.join(self.output_dir, 'condor')))[-1]
        job.write_job_file()
        job.write_executable_file()
        for file_path in [job.job_file, job.executable_file]:
            self.assertTrue(os.path.isfile(file_path))
            read_file_to_output(file_path)
        # submit
        self.assertEqual(job.submit(), 0)
        self.assertTrue(len(str(subprocess.check_output(f'condor_q')).split('\\n')) > 10)
        subprocess.call('condor_q')
        while len(str(subprocess.check_output(f'condor_q')).split('\\n')) > 10:
            time.sleep(1)  # Assuming this is only condor job
        # when finished
        for file_path in [job.output_file, job.error_file, job.log_file]:
            self.assertTrue(os.path.isfile(file_path))
            read_file_to_output(file_path)
        error_file_length = get_file_length(job.error_file)
        self.assertEqual(error_file_length, 0)
        read_file_to_output(os.path.join(self.output_dir, 'condor', condor_dir, 'singularity.output'))
        # extra test for local copy
        self.assertEqual(len(glob(os.path.join(self.output_dir, 'dummy_file_*'))), 3)
        read_file_to_output(pre_existing_file)
        self.assertTrue(os.path.isfile(pre_existing_file))
        self.assertEqual(get_file_length(pre_existing_file), 3)
        self.assertTrue(os.path.isfile(os.path.join(nested_path, 'new_file')))
    #
    # @unittest.skip
    # def test_ros_is_already_running(self):  # NOT WORKING CURRENTLY
    #     # launch first 'ros' job
    #     config_dict = {
    #         'output_path': self.output_dir,
    #         'command': 'python src/condor/test/dummy_ros_script.py',
    #         'use_singularity': True,
    #         'gpus': 0,
    #         'check_if_ros_already_in_use': True,
    #         'green_list': ['ricotta']
    #     }
    #     config = CondorJobConfig().create(config_dict=config_dict)
    #     job = CondorJob(config=config)
    #     job.write_job_file()
    #     job.write_executable_file()
    #     self.assertEqual(job.submit(), 0)
    #     time.sleep(10)
    #     # launch first 'second' job
    #     config_dict = {
    #         'output_path': self.output_dir,
    #         'command': 'python src/condor/test/dummy_python_script.py --config src/condor/test/dummy_config.yml',
    #         'use_singularity': True,
    #         'gpus': 0,
    #         'green_list': ['ricotta'],
    #         'check_if_ros_already_in_use': True,
    #     }
    #     config = CondorJobConfig().create(config_dict=config_dict)
    #     job = CondorJob(config=config)
    #     job.write_job_file()
    #     job.write_executable_file()
    #     self.assertEqual(job.submit(), 0)
    #
    #     while len(str(subprocess.check_output(f'condor_q')).split('\\n')) > 10:
    #         time.sleep(1)  # Assuming this is only condor job
    #
    #     # assert second job was put on hold
    #     read_file_to_output(job.log_file)
    #     with open(job.log_file, 'r') as f:
    #         lines = f.readlines()
    #     self.assertTrue(sum(['hold' in l for l in lines]))

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
