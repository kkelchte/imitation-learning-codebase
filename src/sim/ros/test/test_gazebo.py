import unittest

from src.sim.ros.wrappers.process_wrappers import XpraWrapper, ProcessState, RosWrapper


class TestGazebo(unittest.TestCase):

    def test_launch_and_terminate_xpra(self):
        xpra_process = XpraWrapper()
        self.assertEqual(xpra_process.get_state(), ProcessState.Running)
        xpra_process.terminate()
        self.assertEqual(xpra_process.get_state(), ProcessState.Terminated)

    def test_launch_and_terminate_ros(self):
        ros_process = RosWrapper(config={})
        self.assertEqual(ros_process.get_state(), ProcessState.Running)
        ros_process.terminate()
        self.assertEqual(ros_process.get_state(), ProcessState.Terminated)


if __name__ == '__main__':
    unittest.main()
