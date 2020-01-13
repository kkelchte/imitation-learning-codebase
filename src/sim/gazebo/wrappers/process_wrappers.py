from enum import IntEnum

import asyncio

from src.core.asyncio_helper import run
"""Interface with XPRA application

"""


class ProcessState(IntEnum):
    Running = 0
    Terminated = 1
    Unknown = 2
    Initializing = 3


class ProcessWrapper:

    def __init__(self):
        self._state = ProcessState.Initializing

    def get_state(self) -> ProcessState:
        return self._state

    def terminate(self) -> ProcessState:
        self._state = ProcessState.Terminated
        return self._state


class XpraWrapper(ProcessWrapper):

    def __init__(self):
        super().__init__()
        # TODO add code to start XPRA
        # assert no errors occured on startup

    def terminate(self) -> ProcessState:
        # TODO add code to terminate XPRA
        return super().terminate()
