import unittest

import asyncio
from asyncio import subprocess

from src.sim.gazebo.wrappers.process_wrappers import XpraWrapper, ProcessState


async def launch_subprocess_shell(cmd: str) -> subprocess.Process:
    return await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )


class TestGazebo(unittest.TestCase):

    def test_launch_and_terminate_xpra(self):
        xpra_process = XpraWrapper()
        asyncio.sleep(5)
        self.assertEqual(xpra_process.get_state(), ProcessState.Running)
        xpra_process.terminate()
        self.assertEqual(xpra_process.get_state(), ProcessState.Terminated)


if __name__ == '__main__':
    unittest.main()
