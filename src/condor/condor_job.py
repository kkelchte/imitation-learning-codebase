"""Create a condor job file in an output dir and launch job with job outputs stored in same directory

"""
import copy
import glob
import os
import subprocess
import shlex
from typing import List

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from src.core.config_loader import Config
from src.core.utils import get_date_time_tag


@dataclass_json
@dataclass
class CondorJobConfig(Config):
    command: str = None
    config_file: str = ''
    codebase_dir: str = '/users/visics/kkelchte/code/imitation-learning-codebase'
    cpus: int = 4
    gpus: int = 1
    cpu_mem_gb: int = 17
    disk_mem_gb: int = 52
    nice: bool = False
    wall_time_s: int = 60 * 60 * 3
    gpu_mem_mb: int = 1900
    black_list: List = None
    green_list: List = None
    use_singularity: bool = True
    singularity_file: str = sorted(glob.glob('/users/visics/kkelchte/code/imitation-learning-codebase/'
                                             'rosenvironment/singularity/ros_gazebo_cuda_*.sif'))[-1]
    check_for_ros: bool = False
    save_locally: bool = False

    def post_init(self):  # add default options
        if self.black_list is None:
            del self.black_list
        if self.green_list is None:
            del self.green_list
        if not self.output_path.startswith('/'):
            self.output_path = os.path.join(self.codebase_dir, self.output_path)


class CondorJob:

    def __init__(self, config: CondorJobConfig):
        self._config = config
        self.output_dir = os.path.join(config.output_path, 'condor', get_date_time_tag())
        os.makedirs(self.output_dir)

        if config.config_file != '':
            self._config.command += f' --config {config.config_file}'

        # job & machine specs
        self.specs = {
            'RequestCpus': self._config.cpus,
            'Request_GPUs': self._config.gpus,
            'RequestMemory': self._config.cpu_mem_gb,
            'RequestDisk': self._config.disk_mem_gb,
            'Niceuser': self._config.nice,
            '+RequestWalltime': self._config.wall_time_s,
        }

        # files and directories
        self.initial_dir = self._config.codebase_dir
        self.job_file = os.path.join(self.output_dir, 'job.condor')
        self.executable_file = os.path.join(self.output_dir, 'job.executable')
        self.output_file = os.path.join(self.output_dir, 'job.output')
        self.error_file = os.path.join(self.output_dir, 'job.error')
        self.log_file = os.path.join(self.output_dir, 'job.log')

    def _get_requirements(self) -> str:
        requirements = f'(machine =!= LastRemoteHost)'
        for i in range(6):
            requirements += f' && (target.name =!= LastMatchName{i})'
        if self._config.gpus != 0:
            requirements += f' && (CUDAGlobalMemoryMb >= {self._config.gpu_mem_mb})' \
                            f' && (CUDACapability >= 3.5)'
        if self._config.use_singularity:
            requirements += ' && (HasSingularity)'
        if self._config.black_list is not None:
            for bad_machine in self._config.black_list:
                requirements += f' && (machine != \"{bad_machine}.esat.kuleuven.be\")'
        if self._config.green_list is not None:
            requirements += ' ('
            for good_machine in self._config.green_list:
                requirements += f'(machine == \"{good_machine}.esat.kuleuven.be\") ||'
            requirements = f'{requirements[:-2]})'
        return requirements

    def write_job_file(self):
        with open(self.job_file, 'w') as condor_file:
            condor_file.write('Universe \t = vanilla \n')
            condor_file.write('match_list_length \t = 6 \n')
            condor_file.write('Rank \t = Mips \n')
            condor_file.write("periodic_release = ( HoldReasonCode == 1 && HoldReasonSubCode == 0 ) "
                              "|| HoldReasonCode == 26\n")
            for key, value in self.specs.items():
                condor_file.write(f'{key} \t = {value} \n')
            condor_file.write(f'Requirements = {self._get_requirements()}\n')
            condor_file.write(f'Initial_dir = \t {self.initial_dir}\n')
            condor_file.write(f'Executable = \t {self.executable_file}\n')
            condor_file.write(f'Arguments = \t  \n')
            condor_file.write(f'Log = \t {self.log_file}\n')
            condor_file.write(f'Output = \t {self.output_file}\n')
            condor_file.write(f'Error = \t {self.error_file}\n')
            condor_file.write(f'Notification = Error \n')
            condor_file.write(f'stream_error = True \n')
            condor_file.write(f'stream_output = True \n')
            condor_file.write(f'Queue \n')

        subprocess.call(shlex.split("chmod 711 {0}".format(self.job_file)))

    def _add_check_for_ros_lines(self) -> str:
        #  todo
        raise NotImplementedError

    def _add_lines_to_save_locally(self) -> str:
        #  todo
        raise NotImplementedError

    def _add_lines_to_copy_local_data_back(self) -> str:
        #  todo
        raise NotImplementedError

    def write_executable_file(self):
        with open(self.executable_file, 'w') as executable:
            executable.write('#!/bin/bash\n')
            if self._config.check_for_ros and self._config.use_singularity:
                executable.write(self._add_check_for_ros_lines())
            if self._config.save_locally:
                executable.write(self._add_lines_to_save_locally())
            if self._config.use_singularity:
                executable.write(
                    f"/usr/bin/singularity exec --nv {self._config.singularity_file} "
                    f"{os.path.join(self._config.codebase_dir, 'rosenvironment', 'entrypoint.sh')} "
                    f"{self._config.command} "
                    f">> {os.path.join(self.output_dir, 'singularity.output')}\n")
            else:
                executable.write(f'source {self._config.codebase_dir}/virtualenvironment/venv/bin/activate\n')
                executable.write(f'export PYTHONPATH=$PYTHONPATH/{self._config.codebase_dir}\n')
                executable.write(f'{self._config.command}\n')
            executable.write("retVal=$? \n")
            executable.write("echo \"got exit code $retVal\" \n")
            if self._config.save_locally:
                executable.write(self._add_lines_to_copy_local_data_back())
            executable.write("exit $retVal \n")
        subprocess.call(shlex.split("chmod 711 {0}".format(self.executable_file)))

    def submit(self) -> int:
        return subprocess.call(shlex.split(f'condor_submit {self.job_file}'))
