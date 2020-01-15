import unittest
import time

import subprocess

from src.sim.ros.src.process_wrappers import XpraWrapper, ProcessState, RosWrapper


def count_grep_name(grep_str: str) -> int:
    ps_process = subprocess.Popen(["ps", "-ef"],
                                  stdout=subprocess.PIPE)
    grep_process = subprocess.Popen(["grep", grep_str],
                                    stdin=ps_process.stdout,
                                    stdout=subprocess.PIPE)
    output_string = str(grep_process.communicate()[0])
    processed_output_string = [line for line in output_string.split('\\n') if 'grep' not in line
                               and 'test' not in line and len(line) > len(grep_str)]
    return len(processed_output_string)


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
            'gazebo': 'true',
            'world_name': 'empty_world'
        }
        ros_process = RosWrapper(launch_file='load_ros.launch',
                                 config=config)
        self.assertEqual(ros_process.get_state(), ProcessState.Running)
        time.sleep(5)
        self.assertTrue(count_grep_name('gzserver') >= 1)
        ros_process.terminate()
        xpra_process.terminate()
        self.assertEqual(ros_process.get_state(), ProcessState.Terminated)
        self.assertTrue(count_grep_name('gzserver') == 0)

    def test_launch_and_terminate_turtlebot(self):
        xpra_process = XpraWrapper()
        random_seed = 123
        config = {
            'random_seed': random_seed,
            'gazebo': 'true',
            'world_name': 'empty_world',
            'robot_name': 'turtlebot_sim',
            'turtlebot_sim': 'true'
        }
        ros_process = RosWrapper(launch_file='load_ros.launch',
                                 config=config)
        self.assertEqual(ros_process.get_state(), ProcessState.Running)
        time.sleep(5)
        self.assertTrue(count_grep_name('gzserver') >= 1)
        ros_process.terminate()
        xpra_process.terminate()
        self.assertEqual(ros_process.get_state(), ProcessState.Terminated)
        self.assertTrue(count_grep_name('gzserver') == 0)


if __name__ == '__main__':
    unittest.main()
