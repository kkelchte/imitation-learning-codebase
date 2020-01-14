import unittest
import time

import subprocess
import roslibpy

from src.sim.ros.src.process_wrappers import XpraWrapper, ProcessState, RosWrapper


def grep_name(grep_str: str) -> str:
    ps_process = subprocess.Popen(["ps", "-ef"],
                                  stdout=subprocess.PIPE)
    grep_process = subprocess.Popen(["grep", grep_str],
                                    stdin=ps_process.stdout,
                                    stdout=subprocess.PIPE)
    return str(grep_process.communicate()[0])


class TestRos(unittest.TestCase):

    def test_launch_and_terminate_xpra(self):
        xpra_process = XpraWrapper()
        self.assertEqual(xpra_process.get_state(), ProcessState.Running)
        xpra_process.terminate()
        self.assertEqual(xpra_process.get_state(), ProcessState.Terminated)

    def test_launch_and_terminate_ros(self):
        ros_process = RosWrapper(launch_file='empty_ros.launch',
                                 config={})
        self.assertEqual(ros_process.get_state(), ProcessState.Running)
        ros_process.terminate()
        self.assertEqual(ros_process.get_state(), ProcessState.Terminated)

    def test_launch_and_terminate_gazebo(self):
        xpra_process = XpraWrapper()
        random_seed = 123
        config = {
            'random_seed': random_seed,
            # 'gazebo': 'true',
        }
        ros_process = RosWrapper(launch_file='load_ros.launch',
                                 config=config)
        self.assertEqual(ros_process.get_state(), ProcessState.Running)
        time.sleep(10)
        self.assertTrue(grep_name('gzserver').count('\\n') > 1)
        ros_process.terminate()
        xpra_process.terminate()
        self.assertEqual(ros_process.get_state(), ProcessState.Terminated)
        self.assertTrue(grep_name('gzserver').count('\\n') == 1)


if __name__ == '__main__':
    unittest.main()
