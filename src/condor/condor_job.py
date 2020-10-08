"""Create a condor job file in an output dir and launch job with job outputs stored in same directory

"""
import copy
import glob
import os
import subprocess
import shlex
import time
from typing import List, Optional

from dataclasses import dataclass

import yaml
from dataclasses_json import dataclass_json

from src.core.config_loader import Config
from src.core.utils import get_date_time_tag, get_data_dir
from src.condor.helper_functions import strip_command


@dataclass_json
@dataclass
class CondorJobConfig(Config):
    command: str = None
    config_file: str = ''
    codebase_dir: str = f'{os.environ["HOME"]}/code/imitation-learning-codebase'
    cpus: int = 2
    gpus: int = 0
    cpu_mem_gb: int = 3
    disk_mem_gb: int = 52
    nice: bool = False
    wall_time_s: int = 15 * 60
    gpu_mem_mb: int = 1900
    black_list: Optional[List] = None
    green_list: Optional[List] = None
    use_green_list: bool = False
    use_singularity: bool = False
    singularity_file: str = sorted(glob.glob(f'{os.environ["PWD"]}/rosenvironment/singularity/*.sif'))[-1] if len(sorted(glob.glob(f'{os.environ["PWD"]}/rosenvironment/singularity/*.sif'))) != 0 else ''
    check_if_ros_already_in_use: bool = False
    save_locally: bool = False
    save_before_wall_time: bool = False
    extra_requirements: Optional[str] = None

    def __post_init__(self):
        if self.black_list is None:
            del self.black_list
        if self.green_list is None:
            if self.use_green_list:
                with open(os.path.join(self.codebase_dir, 'src/condor/config/green_list.yml')) as f:
                    data = yaml.load(f, Loader=yaml.FullLoader)
                self.green_list = data['green_list']
            else:
                del self.green_list
        if self.extra_requirements is None:
            del self.extra_requirements

    def post_init(self):  # add default options
        if not self.output_path.startswith('/'):
            self.output_path = os.path.join(get_data_dir(self.codebase_dir), self.output_path)


class CondorJob:

    def __init__(self, config: CondorJobConfig):
        self._config = config

        self.output_dir = os.path.basename(config.config_file).split('.')[0] if config.config_file != '' else \
            f'{get_date_time_tag()}_{strip_command(config.command)}'
        self.output_dir = os.path.join(config.output_path, 'condor', self.output_dir)

        i = 0
        while os.path.isdir(self.output_dir):
            self.output_dir = f'{self.output_dir}_{i}' if i == 0 else '_'.join(self.output_dir.split('_')[:-1]) + '_' + str(i)
            i += 1

        os.makedirs(self.output_dir)

        if config.config_file != '':
            self._config.command += f' --config {config.config_file}'

        # job & machine specs
        self.specs = {
            'RequestCpus': self._config.cpus,
            'Request_GPUs': self._config.gpus,
            'RequestMemory': f'{self._config.cpu_mem_gb} G',
            'RequestDisk': f'{self._config.disk_mem_gb} G',
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

        self.local_home = '/tmp/imitation-learning-codebase'
        self.local_output_path = f'{self.local_home}/{os.path.basename(self.output_dir)}_' \
                                 f'{get_date_time_tag()}' if self._config.save_locally else None
        self._original_output_path = None

    def _get_requirements(self) -> str:
        requirements = f'(machineowner == \"Visics\") && (machine =!= LastRemoteHost) && (OpSysAndVer == \"Fedora32\")'
        for i in range(6):
            requirements += f' && (target.name =!= LastMatchName{i})'
        if self._config.gpus != 0:
            requirements += f' && (CUDAGlobalMemoryMb >= {self._config.gpu_mem_mb})' \
                            f' && (CUDACapability >= 3.7)'
        if self._config.use_singularity:
            requirements += ' && (HasSingularity)'
        if self._config.black_list is not None:
            for bad_machine in self._config.black_list:
                requirements += f' && (machine != \"{bad_machine}.esat.kuleuven.be\")'
        if self._config.green_list is not None:
            requirements += ' && ('
            for good_machine in self._config.green_list:
                requirements += f'(machine == \"{good_machine}.esat.kuleuven.be\") || '
            requirements = f'{requirements[:-3]})'
        if self._config.extra_requirements is not None:
            requirements += f' && {self._config.extra_requirements}'
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

    def _add_check_for_ros_lines(self) -> str:  # NOT WORKING CURRENTLY
        lines = 'ClusterId=$(cat $_CONDOR_JOB_AD | grep ClusterId | head -1 | ' \
                'cut -d \'=\' -f 2 | tail -1 | tr -d [:space:]) \n'
        lines += 'ProcId=$(cat $_CONDOR_JOB_AD | grep ProcId | tail -1 | cut -d \'=\' -f 2 | tr -d [:space:]) \n'
        lines += 'JobStatus=$(cat $_CONDOR_JOB_AD | grep JobStatus | tail -1 | cut -d \'=\' -f 2 | tr -d [:space:]) \n'
        lines += 'RemoteHost=$(cat $_CONDOR_JOB_AD | grep RemoteHost | head -1 | cut -d \'=\' -f 2 ' \
                 '| cut -d \'@\' -f 2 | cut -d \'.\' -f 1) \n'
        lines += 'Command=$(cat $_CONDOR_JOB_AD | grep Cmd | grep kkelchte | head -1 | cut -d \'/\' -f 8) \n'

        #lines += f'if [ -e {self.local_home}/* ] ; then'
        lines += 'while [ $(condor_who | grep kkelchte | wc -l) != 1 ] ; do \n'
        lines += '\t echo found other ros job, so leaving machine $RemoteHost \n'
        lines += '\t ssh opal /usr/bin/condor_hold ${ClusterId}.${ProcId} \n'
        lines += '\t while [ $JobStatus = 2 ] ; do \n'
        lines += '\t \t ssh opal /usr/bin/condor_hold ${ClusterId}.${ProcId} \n'
        lines += '\t \t JobStatus=$(cat $_CONDOR_JOB_AD | grep JobStatus | tail -1 |' \
                 ' cut -d \'=\' -f 2 | tr -d [:space:]) \n'
        lines += '\t \t echo \"[$(date +%F_%H:%M:%S) $Command ] sleeping, status: $JobStatus\" \n'
        lines += '\t \t sleep $(( RANDOM % 30 )) \n'
        lines += '\t done \n'
        lines += '\t echo \"[$(date +%F_%H:%M:%S) $Command ] Put $Command on hold, status: $JobStatus\" \n'
        lines += 'done \n'

        lines += 'echo \"[$(date +%F_%H:%M:%S) $Command ] only $(condor_who | grep kkelchte | wc -l) job is running ' \
                 'on $RemoteHost so continue...\" \n'
        return lines

    def _adjust_commands_config_to_save_locally(self) -> str:
        # copy current config file and adjust output path
        if '--config' not in self._config.command:
            return ''
        config_file = self._config.command.split('--config')[-1].strip()
        with open(config_file, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        self._original_output_path = config_dict['output_path']
        if not self._original_output_path.startswith('/'):
            self._original_output_path = f'{get_data_dir(os.environ["HOME"])}/{self._original_output_path}'
        config_dict['output_path'] = self.local_output_path

        # adjust config if hdf5 files are present and store original_to_new_location_tuples
        original_to_new_location_tuples = []

        def adjust_hdf5_files(hdf5_files: List[str]) -> List[str]:
            hdf5_files = [os.path.join(get_data_dir(os.environ['HOME']), original_hdf5_file)
                          if not original_hdf5_file.startswith('/') else original_hdf5_file
                          for original_hdf5_file in hdf5_files]
            new_hdf5_files = [
                os.path.join(self.local_home, f"{len(original_to_new_location_tuples) + index}.hdf5")
                for index in range(len(hdf5_files))
            ]
            original_to_new_location_tuples.extend(zip(hdf5_files, new_hdf5_files))
            return new_hdf5_files

        # Walk through config and search for hdf5_files tag to adjust
        def search_for_key_and_adjust(config: dict) -> dict:
            for key, value in config.items():
                if key == 'hdf5_files':
                    config[key] = adjust_hdf5_files(value)
                elif isinstance(value, dict):
                    config[key] = search_for_key_and_adjust(value)
            return config
        config_dict = search_for_key_and_adjust(config_dict)

        adjusted_config_file = os.path.join(self.output_dir, 'adjusted_config.yml')
        # store adjust config file in condor dir and make command point to adjust config file
        with open(adjusted_config_file, 'w') as f:
            yaml.dump(config_dict, f)
        self._config.command = f'{self._config.command.split("--config")[0]} --config {adjusted_config_file}'

        # add some extra lines to create new output path and copy hdf5 files
        extra_lines = f'mkdir -p {self.local_output_path} \n'
        if os.path.isdir(f'{self._original_output_path}/torch_checkpoints'):
            extra_lines += f'cp -r {self._original_output_path}/torch_checkpoints {self.local_output_path} \n'
        for original_hdf5_file, new_hdf5_file in original_to_new_location_tuples:
            extra_lines += f'echo copying \"{original_hdf5_file}\" \n'
            extra_lines += f'cp {original_hdf5_file} {new_hdf5_file} \n'
        return extra_lines

    def _add_lines_to_copy_local_data_back(self) -> str:
        lines = f'mkdir -p {self._original_output_path} \n'
        lines += f'cp -r {self.local_output_path}/* {self._original_output_path} \n'
        lines += f'rm -r {self.local_output_path} \n'
        return lines

    def _add_lines_check_wall_time(self) -> str:
        """lines checking the start time with the wall time
        and potentially kill command so local saving is done before end
        """
        lines = "echo 'starting to wait for command to finish' \n"
        lines += "CHECKPID=0 \n"
        lines += f"while [ $CHECKPID -eq 0 " \
                 f"-a $(date +%s) -lt $((STARTTIME + {self._config.wall_time_s} - 5 * 60)) ] ; do " \
                 f"echo $((STARTTIME + {self._config.wall_time_s} - 5 * 60 - $(date +%s))) seconds to live \n "\
                 f"sleep 100 \n kill -0 $PROCESSID >> /dev/null\n CHECKPID=$?\n done\n"
        lines += "if [ $CHECKPID == 0 ] ; then \n" \
                 "kill -9 $PROCESSID \n echo 'kill process before end of walltime' \n " \
                 "fake-call-to-raise-error \n" \
                 "fi \n"
        return lines

    def write_executable_file(self):
        with open(self.executable_file, 'w') as executable:
            executable.write('#!/bin/bash\n')
            executable.write("STARTTIME=$(date +%s)\n")
            if self._config.check_if_ros_already_in_use and self._config.use_singularity:
                executable.write(self._add_check_for_ros_lines())
            if self._config.save_locally:
                executable.write(self._adjust_commands_config_to_save_locally())
            if self._config.use_singularity:
                executable.write(
                    f"/usr/bin/singularity exec --nv {self._config.singularity_file} "
                    f"{os.path.join(self._config.codebase_dir, 'rosenvironment', 'entrypoint.sh')} "
                    f"{self._config.command} "
                    f">> {os.path.join(self.output_dir, 'singularity.output')} 2>&1 "
                    f"{'&' if self._config.save_before_wall_time else ''}\n")
            else:
                executable.write(f'source {self._config.codebase_dir}/virtualenvironment/venv/bin/activate\n')
                executable.write(f'export PYTHONPATH=$PYTHONPATH:{self._config.codebase_dir}\n')
                if 'DATADIR' in os.environ.keys():
                    executable.write(f'export DATADIR={os.environ["DATADIR"]}\n')
                executable.write(f'export HOME={os.environ["HOME"]}\n')
                executable.write(f'{self._config.command} {"&" if self._config.save_before_wall_time else ""}\n')
            if self._config.save_before_wall_time:
                executable.write("PROCESSID=$! \n")
                executable.write(self._add_lines_check_wall_time())
            executable.write("retVal=$? \n")
            executable.write("echo \"got exit code $retVal\" \n")
            executable.write(f"touch {self.output_dir}/FINISHED_$retVal \n")
            if self._config.save_locally:
                executable.write(self._add_lines_to_copy_local_data_back())
            executable.write("exit $retVal \n")
        subprocess.call(shlex.split("chmod 711 {0}".format(self.executable_file)))

    def submit(self) -> int:
        try:
            return subprocess.call(shlex.split(f'condor_submit {self.job_file}'))
        except:
            return -1


def create_jobs_from_job_config_files(job_config_files: List[str],
                                      job_config_object: CondorJobConfig = None) -> List[CondorJob]:
    jobs = []
    for config in job_config_files:
        job_config = copy.deepcopy(job_config_object)
        job_config.config_file = config
        condor_job = CondorJob(config=job_config)
        condor_job.write_job_file()
        condor_job.write_executable_file()
        jobs.append(condor_job)
    return jobs
