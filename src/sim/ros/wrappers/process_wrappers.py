import time
from enum import IntEnum
import os
import subprocess
import shlex

from src.core.logger import cprint

"""Interface with other applications such as 
- xpra
- ros
- ros
"""


class ProcessState(IntEnum):
    Running = 0
    Terminated = 1
    Unknown = 2
    Initializing = 3


class ProcessWrapper:

    def __init__(self, name: str = '', control_string: str = ''):
        self._grace_period = 3
        self._name = name if name else 'default'
        self._control_string = control_string if control_string else self._name
        self._state = ProcessState.Initializing

    def get_state(self) -> ProcessState:
        return self._state

    def _set_terminate_state(self) -> None:
        self._state = ProcessState.Terminated

    def _check_running_process_with_ps(self, control_string: str = '') -> bool:
        ps_process = subprocess.Popen(["ps", "-ef"],
                                      stdout=subprocess.PIPE)
        grep_process = subprocess.Popen(["grep", self._name],
                                        stdin=ps_process.stdout,
                                        stdout=subprocess.PIPE)
        # TODO avoid usage of control string by correctly defining number of expected lines:
        if control_string in str(grep_process.communicate()[0]):
            self._state = ProcessState.Running
            return True
        else:
            self._state = ProcessState.Unknown
            return False

    def _run(self, command: str, check: bool = False) -> bool:
        assert(os.path.exists(command))
        process = subprocess.run(
            shlex.split(f'/bin/sh {command}'),
            capture_output=True
        )
        assert process.returncode == 0
        assert process.stderr == b''
        if check:
            return self._check_running_process_with_ps(control_string=self._control_string)
        return True

    def _terminate_by_name(self, name: str = '', control_string: str = '') -> bool:
        name = name if name else self._name
        control_string = control_string if control_string else self._control_string
        subprocess.run(
            shlex.split(
                f'pkill {name}'
            )
        )
        time.sleep(self._grace_period)
        if self._check_running_process_with_ps(control_string=control_string):
            subprocess.run(
                shlex.split(
                    f'pkill -9 {name}'
                )
            )
            if self._check_running_process_with_ps(control_string=control_string):
                return False
            else:
                return True
        else:
            return True


class XpraWrapper(ProcessWrapper):

    def __init__(self):
        super().__init__(name='xpra',
                         control_string='xorg.conf')
        executable = 'src/sim/ros/scripts/xpra.sh'
        assert self._run(executable, check=True)

    def terminate(self) -> ProcessState:
        if self._terminate_by_name():
            self._set_terminate_state()
        else:
            self._state = ProcessState.Unknown
        self.cleanup()
        return self._state

    def _cleanup(self):
        os.remove('.Xauthority')
        os.remove('.xsession-errors')
        os.remove('.xpra/:100.log.old')
        os.remove('.xpra/Xorg-:100.log.old')
        os.remove('.xpra/run-xpra')
        os.remove('.config/user-dirs.dirs')
        os.remove('.config/user-dirs.locale')


def add_config(config: dict) -> str:
    config_str = ' '
    for key, value in config:
        config_str += f'--{key} {value}'
    return config_str


class RosWrapper(ProcessWrapper):

    def __init__(self, config: dict):
        super().__init__(name='ros')
        executable = 'src/sim/ros/scripts/ros.sh'
        executable += add_config(config)
        assert self._run(executable, check=True)

    def terminate(self) -> ProcessState:
        self._terminate_by_name(name='gzserver')
        self._terminate_by_name(name='roscore')
        return self._state
