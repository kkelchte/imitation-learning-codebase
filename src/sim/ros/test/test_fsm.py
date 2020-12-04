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
from src.sim.ros.test.common_utils import TopicConfig, TestPublisherSubscriber, get_fake_odometry, get_fake_laser_scan

""" Test FSM in take off mode with reward for multiple resets
"""


class TestFsm(unittest.TestCase):

    def setUp(self) -> None:
        self.config = {
            'robot_name': 'test_fsm_robot',
            'fsm': True,
            'fsm_config': 'single_run',
            'world_name': 'test_fsm_world',
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
        self.reset_topic = rospy.get_param('/fsm/reset_topic')
        self.pose_topic = rospy.get_param('/robot/odom_topic')
        self.depth_scan_topic = rospy.get_param('/robot/depth_scan_topic')

        # subscribe to fsm state and reward (fsm's output)
        subscribe_topics = [
            TopicConfig(topic_name=self.state_topic, msg_type="String"),
            TopicConfig(topic_name=self.reward_topic, msg_type="RosReward"),
        ]

        # create publishers for all relevant sensors < FSM
        publish_topics = [
            TopicConfig(topic_name=self.reset_topic, msg_type='Empty'),
        ]
        for sensor in rospy.get_param('/robot/sensors'):
            publish_topics.append(
                TopicConfig(topic_name=eval(f'rospy.get_param("{sensor}_topic")'),
                            msg_type=eval(f'rospy.get_param("{sensor}_type")'))
            )

        self.ros_topic = TestPublisherSubscriber(
            subscribe_topics=subscribe_topics,
            publish_topics=publish_topics
        )

    def test_occasions(self):
        self._test_start_up()
        self._test_delay_evaluation()
        self._test_step()
        self._test_on_collision()
        self._test_goal_reached()
        self._test_out_of_time()

    def _test_start_up(self):
        # FSM should start in unknown state, waiting for reset
        # @ startup (before reset)
        while self.state_topic not in self.ros_topic.topic_values.keys():
            rospy.sleep(0.1)
        self.assertEqual('Unknown', self.ros_topic.topic_values[self.state_topic])
        while self.reward_topic not in self.ros_topic.topic_values.keys():
            rospy.sleep(0.1)
        self.assertEqual(0, self.ros_topic.topic_values[self.reward_topic].reward)
        self.assertEqual('Unknown', self.ros_topic.topic_values[self.reward_topic].termination)

    def _test_delay_evaluation(self):
        self.ros_topic.publishers[self.reset_topic].publish(Empty())
        start_time = rospy.get_time()
        while self.ros_topic.topic_values[self.state_topic] == 'Unknown':
            rospy.sleep(0.01)
        delay_duration = rospy.get_time() - start_time
        self.assertLess(abs(self.delay_evaluation - delay_duration), 0.1)
        self.assertEqual('Running', self.ros_topic.topic_values[self.state_topic])

    def _test_step(self):
        self.ros_topic.publishers[self.pose_topic].publish(Odometry())
        rospy.sleep(0.1)
        self.ros_topic.publishers[self.pose_topic].publish(get_fake_odometry(1))
        while self.ros_topic.topic_values[self.reward_topic].reward == 0:
            rospy.sleep(0.1)
        self.assertEqual(rospy.get_param('/world/reward/step/termination'),
                         self.ros_topic.topic_values[self.reward_topic].termination)
        reward_distance_travelled = rospy.get_param('/world/reward/step/weight/distance_travelled') * 1
        self.assertEqual(reward_distance_travelled,
                         self.ros_topic.topic_values[self.reward_topic].reward)
        self.ros_topic.publishers[self.pose_topic].publish(get_fake_odometry(1, 1))
        while self.ros_topic.topic_values[self.reward_topic].reward == reward_distance_travelled:
            rospy.sleep(0.1)
        reward_distance_travelled = rospy.get_param('/world/reward/step/weight/distance_travelled') * 1.414
        self.assertAlmostEqual(reward_distance_travelled,
                               self.ros_topic.topic_values[self.reward_topic].reward,
                               places=2)

    def _test_on_collision(self):
        self.ros_topic.publishers[self.depth_scan_topic].publish(get_fake_laser_scan([.3] * 360))
        while self.ros_topic.topic_values[self.state_topic].termination != 'Terminated':
            rospy.sleep(0.1)
        self.assertEqual(rospy.get_param('/world/reward/on_collision/weight/constant'),
                         self.ros_topic.topic_values[self.reward_topic].reward)
        self.assertEqual(rospy.get_param('/world/reward/on_collision/termination'),
                         self.ros_topic.topic_values[self.reward_topic].termination)

    def _test_goal_reached(self):
        # reset
        self.ros_topic.publishers[self.reset_topic].publish(Empty())
        while self.ros_topic.topic_values[self.state_topic] == 'Unknown':
            rospy.sleep(0.01)
        self.ros_topic.publishers[self.pose_topic].publish(Odometry())
        rospy.sleep(0.5)
        self.ros_topic.publishers[self.pose_topic].publish(
            get_fake_odometry(2, 2, 1)
        )
        while self.ros_topic.topic_values[self.state_topic].termination != 'Terminated':
            rospy.sleep(0.1)
        self.assertEqual(rospy.get_param('/world/reward/goal_reached/weight/constant'),
                         self.ros_topic.topic_values[self.reward_topic].reward)
        self.assertEqual(rospy.get_param('/world/reward/goal_reached/termination'),
                         self.ros_topic.topic_values[self.reward_topic].termination)

    def _test_out_of_time(self):
        # reset
        self.ros_topic.publishers[self.reset_topic].publish(Empty())
        while self.ros_topic.topic_values[self.state_topic] == 'Unknown':
            rospy.sleep(0.01)
        while self.ros_topic.topic_values[self.state_topic].termination != 'Terminated':
            rospy.sleep(0.1)
        self.assertEqual(rospy.get_param('/world/reward/out_of_time/weight/constant'),
                         self.ros_topic.topic_values[self.reward_topic].reward)
        self.assertEqual(rospy.get_param('/world/reward/out_of_time/termination'),
                         self.ros_topic.topic_values[self.reward_topic].termination)

    def tearDown(self) -> None:
        self._ros_process.terminate()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
