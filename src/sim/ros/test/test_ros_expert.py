import os
import shutil
import time
import unittest

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from src.core.utils import get_filename_without_extension, get_data_dir
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.test.common_utils import TopicConfig, TestPublisherSubscriber


class TestRosExpert(unittest.TestCase):

    def start_test(self) -> None:
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        config = {
            'robot_name': 'turtlebot_sim',
            'world_name': 'test_waypoints',
            'gazebo': False,
            'fsm': False,
            'control_mapping': False,
            'ros_expert': True,
            'output_path': self.output_dir,
            'ros_expert_config_file_path_with_extension': f'src/sim/ros/config/actor/ros_expert.yml'
        }

        # spinoff roslaunch
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=config,
                                       visible=False)

        # subscribe to command control
        self.command_topic = '/actor/ros_expert/cmd_vel'
        subscribe_topics = [
            TopicConfig(topic_name=self.command_topic, msg_type="Twist"),
        ]
        # create publishers for all relevant sensors < sensor expert
        self._depth_topic = rospy.get_param('/robot/depth_scan_topic')
        self._depth_type = rospy.get_param('/robot/depth_scan_type')
        self._pose_topic = rospy.get_param('/robot/odometry_topic')
        self._pose_type = rospy.get_param('/robot/odometry_type')

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
        time.sleep(0.1)
        received_twist: Twist = self.ros_topic.topic_values[self.command_topic]
        return received_twist

    def _test_collision_avoidance_with_laser_scan(self):
        # publish depth scan making robot go left
        scan = LaserScan()
        scan.ranges = [2] * 20 + [5] * 30 + [2] * (360-50)
        twist = self.send_scan_and_read_twist(scan)
        self.assertEqual(twist.angular.z, 1)

        # publish depth scan making robot go right
        scan = LaserScan()
        scan.ranges = [1] * 80 + [2] * (360 - 80)
        twist = self.send_scan_and_read_twist(scan)
        self.assertEqual(twist.angular.z, -1)

        # publish depth scan making robot go straight
        scan = LaserScan()
        scan.ranges = [2] * 360
        twist = self.send_scan_and_read_twist(scan)
        self.assertEqual(twist.angular.z, 0)

    def _test_waypoint_following(self):
        waypoints = rospy.get_param('/world/waypoints')
        odom = Odometry()
        odom.pose.pose.position.x = waypoints[0][0]
        odom.pose.pose.position.y = waypoints[0][1]
        odom.pose.pose.orientation.x = 0
        odom.pose.pose.orientation.y = 0
        odom.pose.pose.orientation.z = 0
        odom.pose.pose.orientation.w = 1
        self.ros_topic.publishers[self._pose_topic].publish(odom)
        time.sleep(1)
        received_twist: Twist = self.ros_topic.topic_values[self.command_topic]
        self.assertEqual(received_twist.angular.z, 0)
        # -30 degrees rotated in yaw => compensate in + yaw
        odom = Odometry()
        odom.pose.pose.position.x = waypoints[0][0]
        odom.pose.pose.position.y = waypoints[0][1]
        odom.pose.pose.orientation.x = 0
        odom.pose.pose.orientation.y = 0
        odom.pose.pose.orientation.z = -0.258819
        odom.pose.pose.orientation.w = 0.9659258
        self.ros_topic.publishers[self._pose_topic].publish(odom)
        time.sleep(1)
        received_twist: Twist = self.ros_topic.topic_values[self.command_topic]
        self.assertEqual(received_twist.angular.z, 1)
        # +30 degrees rotated in yaw => compensate in + yaw
        odom = Odometry()
        odom.pose.pose.position.x = waypoints[0][0]
        odom.pose.pose.position.y = waypoints[0][1]
        odom.pose.pose.orientation.x = 0
        odom.pose.pose.orientation.y = 0
        odom.pose.pose.orientation.z = 0.258819
        odom.pose.pose.orientation.w = 0.9659258
        self.ros_topic.publishers[self._pose_topic].publish(odom)
        time.sleep(1)
        received_twist: Twist = self.ros_topic.topic_values[self.command_topic]
        self.assertEqual(received_twist.angular.z, -1)

    def test_ros_expert(self):
        self.start_test()
        self._test_collision_avoidance_with_laser_scan()
        self._test_waypoint_following()
        self.end_test()

    def end_test(self) -> None:
        self._ros_process.terminate()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
