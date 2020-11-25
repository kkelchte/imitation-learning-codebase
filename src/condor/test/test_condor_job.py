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


def wait_for_job_to_finish(log_file):
    max_duration = 5 * 60
    with open(log_file, 'r') as f:
        log_line = f.readlines()[0]
        f.close()
    job_id = int(log_line.split(' ')[1].split('.')[0][1:])
    stime = time.time()
    while 'nJobStatus = 2' in str(subprocess.check_output(shlex.split(f'condor_q -l {job_id}'))) or \
            'nJobStatus = 1' in str(subprocess.check_output(shlex.split(f'condor_q -l {job_id}'))):
        print('wait for job to finish...')
        time.sleep(3)
        if time.time() - stime > max_duration:
            raise TimeoutError(f'job {job_id} takes too long: {time.time() - stime}s > {max_duration}s.')


class TestCondorJob(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{get_data_dir(os.environ["PWD"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

#    @unittest.skip
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
        wait_for_job_to_finish(job.log_file)

        for file_path in [job.output_file, job.error_file, job.log_file]:
            self.assertTrue(os.path.isfile(file_path))
        error_file_length = len(open(job.error_file, 'r').readlines())
        import pdb
        pdb.set_trace()
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
        wait_for_job_to_finish(job.log_file)

        for file_path in [job.output_file, job.error_file, job.log_file]:
            self.assertTrue(os.path.isfile(file_path))
        read_file_to_output(os.path.join(self.output_dir, 'condor', condor_dir, 'singularity.output'))
        error_file_length = get_file_length(job.error_file)
        self.assertEqual(error_file_length, 0)

    def test_translate_keys_to_string(self):
        self.assertEqual('[\"architecture_config\"][\"random_seed\"]',
                         translate_keys_to_string(['architecture_config', 'random_seed']))
        self.assertEqual('[\"trainer_config\"][\"data_loader_config\"][\"batch_size\"]',
                         translate_keys_to_string(['trainer_config', 'data_loader_config', 'batch_size']))
        self.assertEqual('[\"output_path\"]',
                         translate_keys_to_string(['output_path']))

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
        wait_for_job_to_finish(job.log_file)

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

    @unittest.skip
    def test_ros_is_already_running(self):  # NOT WORKING CURRENTLY
        # launch first 'ros' job
        config_dict = {
            'output_path': self.output_dir,
            'command': 'python src/condor/test/dummy_ros_script.py',
            'use_singularity': False,
            'gpus': 0,
            'wall_time_s': 120,
            'cpus': 1,
            'cpu_mem_gb': 3,
            'disk_mem_gb': 5,
            'check_if_ros_already_in_use': True,
            'green_list': ['jade']
        }
        config = CondorJobConfig().create(config_dict=config_dict)
        job = CondorJob(config=config)
        job.write_job_file()
        job.write_executable_file()
        self.assertEqual(job.submit(), 0)
        time.sleep(10)
        # launch first 'second' job
        config_dict = {
            'output_path': self.output_dir,
            'command': 'python src/condor/test/dummy_python_script.py --config src/condor/test/dummy_config.yml',
            'use_singularity': True,
            'gpus': 0,
            'wall_time_s': 120,
            'green_list': ['ricotta'],
            'check_if_ros_already_in_use': True,
        }
        config = CondorJobConfig().create(config_dict=config_dict)
        job = CondorJob(config=config)
        job.write_job_file()
        job.write_executable_file()
        self.assertEqual(job.submit(), 0)
        wait_for_job_to_finish(job.log_file)
        # assert second job was put on hold
        read_file_to_output(job.log_file)
        with open(job.log_file, 'r') as f:
            lines = f.readlines()
        self.assertTrue(sum(['hold' in l for l in lines]))

    @unittest.skip
    def test_local_hdf5_file(self):
        # create fake hdf5 files
        info_0 = generate_random_dataset_in_raw_data(
            os.path.join(self.output_dir, 'fake_data_0'),
            num_runs=2,
            store_hdf5=True
        )
        info_1 = generate_random_dataset_in_raw_data(
            os.path.join(self.output_dir, 'fake_data_1'),
            num_runs=3,
            store_hdf5=True
        )
        # create experiment config using hdf5 files
        os.makedirs(os.path.join(self.output_dir, 'experiment_output'), exist_ok=True)
        experiment_config = {'output_path': os.path.join(self.output_dir, 'experiment_output'),
                             'fake_key_a': [1, 2, 3],
                             'fake_key_b': {
                                 'fake_key_b_0': 'ok',
                                 'fake_key_b_1': {
                                     'hdf5_files': [
                                         os.path.join(self.output_dir, 'fake_data_0', 'train.hdf5'),
                                         os.path.join(self.output_dir, 'fake_data_1', 'train.hdf5')
                                     ]
                                 }},
                             'fake_key_c': {
                                 'hdf5_files': []
                             },
                             'fake_key_d': {
                                 'hdf5_files': [
                                         os.path.join(self.output_dir, 'fake_data_0', 'validation.hdf5'),
                                         os.path.join(self.output_dir, 'fake_data_1', 'validation.hdf5')
                                 ]
                             }}
        original_sizes = [int(subprocess.getoutput("stat --format %s "+v))
                          for v in [os.path.join(self.output_dir, 'fake_data_0', 'train.hdf5'),
                                    os.path.join(self.output_dir, 'fake_data_1', 'train.hdf5'),
                                    os.path.join(self.output_dir, 'fake_data_0', 'validation.hdf5'),
                                    os.path.join(self.output_dir, 'fake_data_1', 'validation.hdf5')]]
        print(f'original_sizes: {original_sizes}')

        with open(os.path.join(self.output_dir, 'experiment_config.yml'), 'w') as f:
            yaml.dump(experiment_config, f)

        # create and submit condor job using hdf5 files but save them locally
        job_config = {
            'config_file': os.path.join(self.output_dir, 'experiment_config.yml'),
            'output_path': self.output_dir,
            'command': 'python src/condor/test/dummy_python_script_check_hdf5.py',
            'wall_time_s': 60,
            'save_locally': True
        }
        condor_job = CondorJob(config=CondorJobConfig().create(config_dict=job_config))
        condor_job.write_job_file()
        condor_job.write_executable_file()
        condor_job.submit()
        wait_for_job_to_finish(condor_job.log_file)

        self.assertTrue(glob(os.path.join(condor_job.output_dir, 'FINISHED*'))[0].endswith('0'))
        # check jobs output file to control hdf5 files were loaded locally
        # compare file sizes to ensure same hdf5 files were copied
        with open(os.path.join(condor_job.output_dir, 'job.output'), 'r') as f:
            output_lines = f.readlines()

        hdf5_files = [l.split(' ')[1] for l in output_lines if l.startswith('HDF5_FILE')]
        hdf5_file_sizes = [int(l.split(' ')[2]) for l in output_lines if l.startswith('HDF5_FILE')]
        print(f'hdf5_files: {hdf5_files}')
        print(f'hdf5_file_sizes: {hdf5_file_sizes}')
        self.assertEqual(len(hdf5_file_sizes), len(original_sizes))
        for f in hdf5_files:
            self.assertTrue(f.startswith(condor_job.local_home))
        for js, rs in zip(hdf5_file_sizes, original_sizes):
            self.assertEqual(js, rs)

    @unittest.skip
    def test_stop_before_wall_time(self):
        config_dict = {
            'output_path': self.output_dir,
            'command': 'python src/condor/test/dummy_python_script.py',
            'use_singularity': True,
            'save_locally': True,
            'save_before_wall_time': True,
            'wall_time_s': 60,
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

        self.assertEqual(job.submit(), 0)
        wait_for_job_to_finish(job.log_file)

        for file_path in [job.output_file, job.error_file, job.log_file]:
            self.assertTrue(os.path.isfile(file_path))
        self.assertTrue(os.path.isfile(os.path.join(self.output_dir, 'condor', condor_dir, 'FINISHED_127')))

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)
        pass


if __name__ == '__main__':
    unittest.main()
