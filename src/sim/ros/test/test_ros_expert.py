import time
import unittest

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.test.common_utils import TopicConfig, TestPublisherSubscriber


class TestRosExpert(unittest.TestCase):

    def start_test(self, config: str) -> None:
        config = {
            'robot_name': 'turtle_sim',
            'gazebo': False,
            'fsm': False,
            'control_mapper': False,
            'ros_expert': True,
            'expert_config_file_path_with_extension': f'src/sim/ros/config/actor/{config}.yml'
        }

        # spinoff roslaunch
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=config,
                                       visible=True)

        # subscribe to command control
        self.command_topic = rospy.get_param('/actor/ros_expert')
        subscribe_topics = [
            TopicConfig(topic_name=self.command_topic, msg_type="Twist"),
        ]
        # create publishers for all relevant sensors < sensor expert
        self._depth_topic = rospy.get_param('/robot/depth_scan_topic')
        self._depth_type = rospy.get_param('/robot/depth_scan_type')
        self._pose_topic = rospy.get_param('/robot/pose_estimation_topic')
        self._pose_type = rospy.get_param('/robot/pose_estimation_type')

        publish_topics = [
            TopicConfig(topic_name=self._depth_topic, msg_type=self._depth_type),
            TopicConfig(topic_name=self._pose_topic, msg_type=self._pose_type)
        ]

        self.ros_topic = TestPublisherSubscriber(
            subscribe_topics=subscribe_topics,
            publish_topics=publish_topics
        )

    def send_scan_and_read_twist(self, scan: LaserScan) -> Twist:
        self.ros_topic.publishers[self._depth_topic].publish(scan)
        time.sleep(1)
        received_twist: Twist = self.ros_topic.topic_values[self.command_topic]
        return received_twist.angular.z

    def _test_collision_avoidance(self):
        # publish depth scan making robot go left
        scan = LaserScan()
        scan.ranges = [2]*360
        scan.ranges[50:55] = 5
        twist = self.send_scan_and_read_twist(scan)
        self.assertTrue(twist.angular.z == 1)

        # publish depth scan making robot go right
        scan = LaserScan()
        scan.ranges = [2] * 360
        scan.ranges[:80] = 1
        twist = self.send_scan_and_read_twist(scan)
        self.assertTrue(twist.angular.z == -1)

        # publish depth scan making robot go straight
        scan = LaserScan()
        scan.ranges = [2] * 360
        twist = self.send_scan_and_read_twist(scan)
        self.assertTrue(twist.angular.z == 0)

    def test_ros_expert(self):
        self.start_test(config='default')
        self._test_collision_avoidance()


if __name__ == '__main__':
    unittest.main()
