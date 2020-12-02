import os
import shutil
import unittest
import time
from typing import List

import rospy
from dataclasses import dataclass

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty, Float32, String  # Do not remove!
from sensor_msgs.msg import LaserScan  # Do not remove!

from imitation_learning_ros_package.msg import RosReward

from src.core.utils import get_filename_without_extension, get_data_dir
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.core.data_types import TerminationType
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.test.common_utils import TopicConfig, TestPublisherSubscriber

""" Test FSM in take off mode with reward for multiple resets
"""


class TestFsm(unittest.TestCase):

    def setUp(self) -> None:
        self.config = {
            'robot_name': 'dummy',
            'fsm': True,
            'fsm_config': 'single_run',
            'world_name': 'debug_drone',
            'control_mapping': False,
            'waypoint_indicator': False,
        }
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        self.config['output_path'] = self.output_dir

        # spinoff roslaunch
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=self.config,
                                       visible=True)
        self.delay_evaluation = rospy.get_param('/world/delay_evaluation')
        self.state_topic = rospy.get_param('/fsm/state_topic')
        self.reward_topic = rospy.get_param('/fsm/reward_topic')

        # subscribe to fsm state and reward (fsm's output)
        subscribe_topics = [
            TopicConfig(topic_name=self.state_topic, msg_type="String"),
            TopicConfig(topic_name=self.reward_topic, msg_type="RosReward"),
        ]
        # create publishers for all relevant sensors < FSM
        self._pose_topic = rospy.get_param('/robot/odometry_topic')
        self._pose_type = rospy.get_param('/robot/odometry_type')
        self._reset_topic = rospy.get_param('/fsm/reset_topic', '/fsm/reset')
        publish_topics = [
            TopicConfig(topic_name=self._pose_topic, msg_type=self._pose_type),
            TopicConfig(topic_name=self._reset_topic, msg_type='Empty'),

        ]

        self.ros_topic = TestPublisherSubscriber(
            subscribe_topics=subscribe_topics,
            publish_topics=publish_topics
        )

    def test_run_and_reach_goal(self):
        # FSM should start in unknown state, waiting for reset
        # @ startup (before reset)
        while self.state_topic not in self.ros_topic.topic_values.keys():
            rospy.sleep(0.1)
        self.assertEqual('Unknown', self.ros_topic.topic_values[self.state_topic])
        while self.reward_topic not in self.ros_topic.topic_values.keys():
            rospy.sleep(0.1)
        self.assertEqual(0, self.ros_topic.topic_values[self.reward_topic].reward)
        self.assertEqual('Unknown', self.ros_topic.topic_values[self.reward_topic].termination)

        for _ in range(2):
            print(f'{rospy.get_time()}: iteration {_}')
            # @ reset -> delay evaluation -> running
            self.ros_topic.publishers[self._pose_topic].publish(Odometry())
            self.ros_topic.publishers[self._reset_topic].publish(Empty())
            # @ second iteration: wait for reset to be received at fsm
            while self.ros_topic.topic_values[self.state_topic] == 'Terminated':
                rospy.sleep(0.01)
            start_time = rospy.get_time()
            while self.ros_topic.topic_values[self.state_topic] == 'Unknown':
                rospy.sleep(0.01)
            delay_duration = rospy.get_time() - start_time
            self.assertLess(abs(self.delay_evaluation - delay_duration), 0.1)

            self.assertEqual('Running', self.ros_topic.topic_values[self.state_topic])
            self.assertEqual('NotDone', self.ros_topic.topic_values[self.reward_topic].termination)
            self.assertEqual(-1, self.ros_topic.topic_values[self.reward_topic].reward)

            # FSM should @ max distance crossed switch to terminated state and return finish success and big reward
            odom = Odometry()
            odom.pose.pose.position.x = 2
            odom.pose.pose.position.y = 2
            odom.pose.pose.position.z = 1
            self.ros_topic.publishers[self._pose_topic].publish(odom)
            max_duration = 5
            start_time = rospy.get_time()
            while self.ros_topic.topic_values[self.reward_topic].termination == "NotDone" and \
                    rospy.get_time() - start_time < max_duration:
                rospy.sleep(0.5)
            self.assertLess(rospy.get_time() - start_time, max_duration)
            self.assertEqual('Terminated', self.ros_topic.topic_values[self.state_topic])
            self.assertEqual('Success', self.ros_topic.topic_values[self.reward_topic].termination)
            self.assertEqual(100, self.ros_topic.topic_values[self.reward_topic].reward)

    def tearDown(self) -> None:
        self._ros_process.terminate()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
