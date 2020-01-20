#!/usr/bin/python3.7
import unittest
import time

import subprocess
import roslib
import numpy as np

from src.sim.common.actors import ActorConfig
from src.sim.common.data_types import TerminalType, EnvironmentType, Action, ActorType, State
from src.sim.common.environment import EnvironmentConfig, RosConfig, RosLaunchConfig
from src.sim.ros.src.process_wrappers import ProcessState, RosWrapper
from src.sim.ros.src.ros_actors import RosExpert
from src.sim.ros.src.ros_environment import RosEnvironment
roslib.load_manifest('imitation_learning_ros_package')
import rospy


def count_grep_name(grep_str: str) -> int:
    ps_process = subprocess.Popen(["ps", "-ef"],
                                  stdout=subprocess.PIPE)
    with ps_process.stdout:
        grep_process = subprocess.Popen(["grep", grep_str],
                                        stdin=ps_process.stdout,
                                        stdout=subprocess.PIPE)
        with grep_process.stdout:
            output_string = str(grep_process.communicate()[0])
    processed_output_string = [line for line in output_string.split('\\n') if 'grep' not in line
                               and 'test' not in line and len(line) > len(grep_str) and 'pycharm' not in line]
    return len(processed_output_string)


class TestRos(unittest.TestCase):

    # def test_launch_and_terminate_xpra_DEPRECATED(self):
    #     xpra_process = XpraWrapper()
    #     self.assertEqual(xpra_process.get_state(), ProcessState.Running)
    #     xpra_process.terminate()
    #     self.assertEqual(xpra_process.get_state(), ProcessState.Terminated)

    @unittest.skip
    def test_launch_and_terminate_ros(self):
        ros_process = RosWrapper(launch_file='empty_ros.launch',
                                 config={})
        self.assertEqual(ros_process.get_state(), ProcessState.Running)
        self.assertTrue(count_grep_name('ros') > 0)
        ros_process.terminate()
        self.assertEqual(ros_process.get_state(), ProcessState.Terminated)

    @unittest.skip
    def test_launch_and_terminate_gazebo(self):
        random_seed = 123
        config = {
            'random_seed': random_seed,
            'gazebo': 'true',
            'world_name': 'empty_world'
        }
        ros_process = RosWrapper(launch_file='load_ros.launch',
                                 config=config,
                                 visible=True)
        self.assertEqual(ros_process.get_state(), ProcessState.Running)
        time.sleep(5)
        self.assertTrue(count_grep_name('gzserver') >= 1)
        ros_process.terminate()
        self.assertEqual(ros_process.get_state(), ProcessState.Terminated)
        self.assertTrue(count_grep_name('gzserver') == 0)

    @unittest.skip
    def test_launch_and_terminate_turtlebot_with_keyboard_navigation(self):
        duration_min = 0.2
        random_seed = 123
        config = {
            'random_seed': random_seed,
            'gazebo': 'true',
            'graphics': 'true',
            'world_name': 'empty_world',
            'robot_name': 'turtlebot_sim',
            'turtlebot_sim': 'true'
        }
        ros_process = RosWrapper(config=config)
        self.assertEqual(ros_process.get_state(), ProcessState.Running)
        time.sleep(int(duration_min * 60))
        self.assertEqual(rospy.get_param('command_topic'), '/cmd_vel')
        self.assertTrue(count_grep_name('gzserver') >= 1)
        ros_process.terminate()
        self.assertEqual(ros_process.get_state(), ProcessState.Terminated)
        self.assertTrue(count_grep_name('gzserver') == 0)

    @unittest.skip
    def test_image_view_from_turtlebot(self):
        duration_min = 0.5
        random_seed = 123
        config = {
            'random_seed': random_seed,
            'gazebo': 'true',
            'graphics': 'false',
            'world_name': 'empty_world',
            'robot_name': 'turtlebot_sim',
            'turtlebot_sim': 'true'
        }
        ros_process = RosWrapper(config=config)
        self.assertEqual(ros_process.get_state(), ProcessState.Running)
        time.sleep(int(duration_min * 60))
        self.assertTrue(count_grep_name('gzserver') >= 1)
        ros_process.terminate()
        self.assertEqual(ros_process.get_state(), ProcessState.Terminated)
        self.assertTrue(count_grep_name('gzserver') == 0)

    def test_ros_environment(self):
        # spinoff RosEnvironment with config containing turtlebot and images
        config = EnvironmentConfig(
            factory_key=EnvironmentType.Ros,
            max_number_of_steps=10,
            ros_config=RosConfig(
                visible_xterm=True,
                ros_launch_config=RosLaunchConfig(
                    random_seed=123,
                    gazebo=True,
                    world_name='object_world',
                    robot_name='turtlebot_sim',
                    turtlebot_sim=True
                )
            )
        )
        environment = RosEnvironment(config=config)
        # take a 10 steps
        state = environment.reset()
        while state.terminal == TerminalType.NotDone:
            print(f'State: {state.terminal}: {state.time_stamp_us} \n'
                  f'depth: {state.sensor_data["depth_scan"]}')
            if True:
                import matplotlib.pyplot as plt
                plt.imshow(state.sensor_data['forward_camera'])
                plt.show()
            action = Action(
                actor_type=ActorType.Expert,
                value=np.array((1, 0, 0, 0, 0, 1))
            )
            state = environment.step(action)

    def test_ros_actor(self):
        config = ActorConfig(
            description='basic expert',
            actor_type=ActorType.Expert,
            actor_specs={
                'sensor_name': 'depth_scan'
            }
        )
        actor = RosExpert(config=config)
        sensor_data = {
            'forward_camera': np.zeros((128, 128, 3)),
            'depth_scan': np.zeros((90, 1))
        }
        action = actor.get_action(sensor_data=sensor_data)
        self.assertTrue(isinstance(action, Action))



if __name__ == '__main__':
    unittest.main()
