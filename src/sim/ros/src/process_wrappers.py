import shutil
import time
from enum import IntEnum
import os
import subprocess
import shlex
from datetime import datetime

from src.core.logger import cprint

"""Interface with other applications such as 
- xpra
- ros
"""


class ProcessState(IntEnum):
    Running = 0
    Terminated = 1
    Unknown = 2
    Initializing = 3


class ProcessWrapper:

    def __init__(self,
                 name: str = '',
                 grep_str: str = ''):
        self._grace_period = 3
        self._name = name if name else 'default'
        self._state = ProcessState.Initializing
        self._grep_str = grep_str if grep_str else self._name
        self._control_str = ''
        self._process_popen = None

    def get_state(self) -> ProcessState:
        return self._state

    def _check_running_process_with_ps(self,
                                       grep_str: str = '') -> bool:
        grep_str = grep_str if grep_str else self._grep_str
        ps_process = subprocess.Popen(["ps", "-ef"],
                                      stdout=subprocess.PIPE)
        grep_process = subprocess.Popen(["grep", grep_str],
                                        stdin=ps_process.stdout,
                                        stdout=subprocess.PIPE)
        output_string = str(grep_process.communicate()[0])
        processed_output_string = [line for line in output_string.split('\\n') if 'grep' not in line
                                   and 'test' not in line and len(line) > len(grep_str)]
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
        if self._check_running_process_with_ps():
            self._state = ProcessState.Running
            return True
        else:
            self._state = ProcessState.Unknown
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

    def _terminate_by_pid(self):
        if self._process_popen.poll() is None:
            self._process_popen.terminate()
            self._process_popen.wait()

    def terminate(self) -> ProcessState:
        if self._terminate_by_name():
            self._state = ProcessState.Terminated
        else:
            self._state = ProcessState.Unknown
        self._cleanup()
        return self._state

    def _cleanup(self):
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
            config_str += f" {key}:=\'{value}\'"
        else:
            config_str += f' {key}:={value}'
    return config_str


class RosWrapper(ProcessWrapper):

    def __init__(self, launch_file: str, config: dict, visible: bool = False):
        super().__init__(name='ros')
        executable = 'src/sim/ros/scripts/ros.sh'
        launch_file = os.path.join(os.environ['HOME'], 'src', 'sim', 'ros', 'catkin_ws', 'src',
                                   'imitation_learning_ros_package', 'launch', launch_file)
        assert os.path.isfile(launch_file)
        executable += ' ' + launch_file
        executable += adapt_launch_config(config)
        command = f'xterm -iconic -l -lf "{os.environ["HOME"]}/.ros/ '\
                  f'{datetime.strftime(datetime.now(), format="%y-%m-%d_%H:%M:%S")}_xterm_output" '\
                  f'-hold -e {executable}'
        if not visible:
            command = f'xvfb-run -a {command}'
        assert self._run(command,
                         strict_check=False,
                         shell=False,
                         background=True)
        # TODO pipe stderr of ROS to logger debug.

    @staticmethod
    def _cleanup():
        shutil.rmtree('.ros', ignore_errors=True)
        shutil.rmtree('.gazebo', ignore_errors=True)

    def terminate(self) -> ProcessState:
        self._terminate_by_pid()
        if self._terminate_by_name(command_name='gz') and \
            self._terminate_by_name(command_name='xterm'):
            self._state = ProcessState.Terminated
        else:
            self._state = ProcessState.Unknown
        self._cleanup()
        return self._state


