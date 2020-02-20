import shutil
import time
from enum import IntEnum
import os
import subprocess
import shlex
from datetime import datetime

import rospy

from src.core.logger import cprint, MessageType, get_logger
from src.sim.common.data_types import ProcessState

"""Interface with other applications such as 
- xpra
- ros
"""


class ProcessWrapper:

    def __init__(self,
                 name: str = '',
                 grep_str: str = ''):
        self._grace_period = 1
        self._name = name if name else 'default'
        self._state = ProcessState.Initializing
        self._grep_str = grep_str if grep_str else self._name
        self._control_str = ''
        self._process_popen = None
        self._logger = get_logger(name=__name__)
        cprint(f'initiate', self._logger)

    def get_state(self) -> ProcessState:
        return self._state

    def _check_running_process_with_ps(self,
                                       grep_str: str = '') -> bool:
        grep_str = grep_str if grep_str else self._grep_str
        ps_process = subprocess.Popen(["ps", "-ef"],
                                      stdout=subprocess.PIPE)
        with ps_process.stdout:
            grep_process = subprocess.Popen(["grep", grep_str],
                                            stdin=ps_process.stdout,
                                            stdout=subprocess.PIPE)
            with grep_process.stdout:
                output_string = str(grep_process.communicate()[0])
        self._terminate_by_pid(process=ps_process)
        self._terminate_by_pid(process=grep_process)
        processed_output_string = [line for line in output_string.split('\\n') if 'grep' not in line
                                   and 'test' not in line and len(line) > len(grep_str) and 'pycharm' not in line]
        return len(processed_output_string) >= 1

    def _run(self, command: str, strict_check: bool = False, shell: bool = False, background: bool = True) -> bool:
        if shell:
            assert(os.path.exists(command.split(' ')[0]))
            command = f'/bin/bash {command}'
        if background:
            self._process_popen = subprocess.Popen(shlex.split(command))
        else:
            process = subprocess.run(
                shlex.split(command),
                capture_output=True
            )
            if strict_check:
                assert process.returncode == 0
                assert process.stderr == b''
        stime = time.time()
        max_duration = 30
        while not self._check_running_process_with_ps() and (time.time() - stime) < max_duration:
            time.sleep(0.1)
        if self._check_running_process_with_ps():
            self._state = ProcessState.Running
            return True
        else:
            self._state = ProcessState.Unknown
            self.terminate()
            return False

    def _terminate_by_name(self, command_name: str = '') -> bool:
        command_name = command_name if command_name else self._name
        subprocess.run(
            shlex.split(
                f'pkill {command_name}'
            )
        )
        time.sleep(self._grace_period)
        if self._check_running_process_with_ps(grep_str=command_name):
            subprocess.run(
                shlex.split(
                    f'pkill -9 {command_name}'
                )
            )
            time.sleep(self._grace_period)
            if self._check_running_process_with_ps(grep_str=command_name):
                return False
            else:
                return True
        else:
            return True

    def _terminate_by_pid(self, process: subprocess.Popen = None):
        process = process if process is not None else self._process_popen
        if process is None:
            return
        max_duration = 30
        start_time = time.time()
        while process.poll() is None and time.time() - start_time < max_duration:
            process.terminate()
            process.wait()
            time.sleep(1)

    def terminate(self) -> ProcessState:
        if self._terminate_by_name():
            self._state = ProcessState.Terminated
        else:
            self._state = ProcessState.Unknown
        self._cleanup()
        return self._state

    @staticmethod
    def _cleanup():
        pass


class XpraWrapper(ProcessWrapper):

    def __init__(self):
        super().__init__(
            name='xpra',
            grep_str='/usr/bin/xpra'
        )
        executable = 'src/sim/ros/scripts/xpra.sh'
        assert self._run(executable,
                         strict_check=True,
                         background=False,
                         shell=True)

    @staticmethod
    def _cleanup():
        shutil.rmtree('.nv', ignore_errors=True)
        shutil.rmtree('.xpra', ignore_errors=True)


def adapt_launch_config(config: dict) -> str:
    config_str = ''
    for key, value in config.items():
        if isinstance(value, str):
            config_str += f" {key}:={value}"
        elif isinstance(value, bool):
            config_str += f" {key}:=\'true\'" if value else f" {key}:=\'false\'"
        else:
            config_str += f' {key}:={value}'
    return config_str


class RosWrapper(ProcessWrapper):

    def __init__(self, config: dict, launch_file: str = 'load_ros.launch', visible: bool = False):
        super().__init__(name='ros')
        post_init_delay = 4
        self._grace_period = 3

        # executable = os.path.join(os.environ['HOME'], 'src', 'sim', 'ros', 'scripts', 'ros_DEPRECATED.sh')
        executable = 'roslaunch '
        if not visible:
            executable = f'xvfb-run -a {executable}'
        launch_file = os.path.join(os.environ['HOME'], 'src', 'sim', 'ros', 'catkin_ws', 'src',
                                   'imitation_learning_ros_package', 'launch', launch_file)
        assert os.path.isfile(launch_file)
        executable += ' ' + launch_file
        executable += adapt_launch_config(config)
        if not os.path.isdir(f'{os.environ["HOME"]}/.ros/'):
            os.makedirs(f'{os.environ["HOME"]}/.ros/')
        command = f'env -u SESSION_MANAGER xterm -iconic -l -lf "{os.environ["HOME"]}/.ros/'\
                  f'{datetime.strftime(datetime.now(), format="%y-%m-%d_%H:%M:%S")}_xterm_output.log" '\
                  f'-hold -e {executable}'
        if not visible:
            command = f'xvfb-run -a {command}'
        assert self._run(command,
                         strict_check=False,
                         shell=False,
                         background=True)
        if 'gazebo' in config.keys() and config['gazebo'] == 'true':
            while not self._check_running_process_with_ps('gzserver'):
                time.sleep(1)
        success = False  # wait for ros server to be started by providing params
        while not success:
            try:
                rospy.has_param('output_path')
            except:
                time.sleep(0.1)
            else:
                success = True
        time.sleep(post_init_delay)

        # TODO pipe stderr of ROS to logger debug.

    @staticmethod
    def _cleanup():
        shutil.rmtree('.ros', ignore_errors=True)
        shutil.rmtree('.gazebo', ignore_errors=True)

    def _all_terminated_by_name(self, *args):
        outcomes = [
            self._terminate_by_name(command_name=argument) for argument in args
        ]
        cprint(f'termination outcomes: {outcomes}', self._logger, msg_type=MessageType.debug)
        return sum(outcomes) == len(outcomes)

    def terminate(self) -> ProcessState:
        self._terminate_by_pid()

        start_time = time.time()
        max_duration = 20
        while not self._all_terminated_by_name('ros', 'gz', 'xterm', 'xvfb') \
                and time.time() - start_time < max_duration:
            time.sleep(1)
        if time.time() - start_time < max_duration:
            self._state = ProcessState.Terminated
        else:
            self._state = ProcessState.Unknown
        self._cleanup()
        return self._state
