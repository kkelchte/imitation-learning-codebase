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

from src.core.utils import get_filename_without_extension
from src.sim.ros.catkin_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.core.data_types import TerminationType
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.test.common_utils import TopicConfig, TestPublisherSubscriber

""" Test FSM in take off mode with reward for multiple resets
"""


class TestFsm(unittest.TestCase):

    def setUp(self) -> None:
        self.config = {
            'robot_name': 'drone_sim',
            'fsm': True,
            'fsm_config': 'takeoff_run',
            'world_name': 'debug',
            'control_mapping': False,
            'waypoint_indicator': False,
        }
        self.output_dir = f'test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        self.config['output_path'] = self.output_dir

        # spinoff roslaunch
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=self.config,
                                       visible=False)
        self.delay_evaluation = rospy.get_param('/world/delay_evaluation')
        self.state_topic = rospy.get_param('/fsm/state_topic')
        self.reward_topic = rospy.get_param('/fsm/reward_topic')

        # subscribe to fsm state and reward (fsm's output)
        subscribe_topics = [
            TopicConfig(topic_name=self.state_topic, msg_type="String"),
            TopicConfig(topic_name=self.reward_topic, msg_type="RosReward"),
        ]
        # create publishers for all relevant sensors < FSM
        self._depth_topic = rospy.get_param('/robot/depth_scan_topic')
        self._depth_type = rospy.get_param('/robot/depth_scan_type')
        self._pose_topic = rospy.get_param('/robot/odometry_topic')
        self._pose_type = rospy.get_param('/robot/odometry_type')
        self._reset_topic = rospy.get_param('/fsm/reset_topic', '/reset')
        publish_topics = [
            TopicConfig(topic_name=self._depth_topic, msg_type=self._depth_type),
            TopicConfig(topic_name=self._pose_topic, msg_type=self._pose_type),
            TopicConfig(topic_name=self._reset_topic, msg_type='Empty'),

        ]

        self.ros_topic = TestPublisherSubscriber(
            subscribe_topics=subscribe_topics,
            publish_topics=publish_topics
        )

    def test_take_off_run_and_reach_goal(self):
        # FSM should start in unknown state, waiting for reset
        rospy.sleep(0.5)
        self.assertEqual('Unknown', self.ros_topic.topic_values[self.state_topic])
        self.assertEqual(0, self.ros_topic.topic_values[self.reward_topic].reward)
        self.assertEqual('Unknown', self.ros_topic.topic_values[self.reward_topic].termination)
        for _ in range(3):
            # FSM should @ reset wait delay evaluation and then take off
            odom = Odometry()
            self.ros_topic.publishers[self._pose_topic].publish(odom)
            self.ros_topic.publishers[self._reset_topic].publish(Empty())
            rospy.sleep(self.delay_evaluation + 0.5)
            self.assertEqual('TakeOff', self.ros_topic.topic_values[self.state_topic])
            self.assertEqual(0, self.ros_topic.topic_values[self.reward_topic].reward)
            self.assertEqual('Unknown', self.ros_topic.topic_values[self.reward_topic].termination)
            # FSM should @ correct height switch to running state with step reward and NotDone termination
            odom = Odometry()
            odom.pose.pose.position.z = 0.2
            self.ros_topic.publishers[self._pose_topic].publish(odom)
            max_duration = 5
            start_time = rospy.get_time()
            while self.ros_topic.topic_values[self.state_topic] == "TakeOff" and \
                rospy.get_time() - start_time < max_duration:
                rospy.sleep(0.5)
            self.assertLess(rospy.get_time() - start_time, max_duration)
            self.assertEqual('Running', self.ros_topic.topic_values[self.state_topic])
            self.assertEqual('NotDone', self.ros_topic.topic_values[self.reward_topic].termination)
            self.assertEqual(-1, self.ros_topic.topic_values[self.reward_topic].reward)
            # FSM should @ max distance crossed switch to terminated state and return finish success and big reward
            odom = Odometry()
            odom.pose.pose.position.x = 1.5
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
