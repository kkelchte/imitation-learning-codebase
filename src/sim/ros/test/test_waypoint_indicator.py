import os
import shutil
import time
import unittest

import numpy as np
import rospy
from nav_msgs.msg import Odometry

from src.core.utils import get_filename_without_extension, get_data_dir
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.test.common_utils import TopicConfig, TestPublisherSubscriber


class TestWaypointIndicator(unittest.TestCase):

    def start_test(self) -> None:
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

        config = {
            'output_path': self.output_dir,
            'world_name': 'test_waypoints',
            'robot_name': 'turtlebot_sim',
            'gazebo': False,
            'fsm': False,
            'control_mapping': False,
            'ros_expert': False,
            'waypoint_indicator': True
        }

        # spinoff roslaunch
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=config,
                                       visible=False)

        # subscribe to command control
        self._waypoint_topic = '/waypoint_indicator/current_waypoint'
        subscribe_topics = [
            TopicConfig(topic_name=self._waypoint_topic, msg_type="Float32MultiArray"),
        ]
        # create publishers for all relevant sensors < sensor expert
        self._pose_topic = rospy.get_param('/robot/position_sensor/topic')
        self._pose_type = rospy.get_param('/robot/position_sensor/type')

        publish_topics = [
            TopicConfig(topic_name=self._pose_topic, msg_type=self._pose_type)
        ]

        self.ros_topic = TestPublisherSubscriber(
            subscribe_topics=subscribe_topics,
            publish_topics=publish_topics
        )

    def send_odom_and_read_next_waypoint(self, odom: Odometry) -> tuple:
        self.ros_topic.publishers[self._pose_topic].publish(odom)
        time.sleep(1)
        received_waypoint: tuple = self.ros_topic.topic_values[self._waypoint_topic].data
        return received_waypoint

    def compare_vectors(self, a: tuple, b: tuple) -> bool:
        return sum(np.asarray(a) - np.asarray(b)) < 10 ** -3

    def _test_first_waypoint(self):
        odom = Odometry()
        odom.pose.pose.position.x = -31
        odom.pose.pose.position.y = -5
        received_waypoint = self.send_odom_and_read_next_waypoint(odom=odom)
        self.assertTrue(self.compare_vectors(received_waypoint, self.waypoints[0]))

    def _test_transition_of_waypoint(self):
        odom = Odometry()
        odom.pose.pose.position.x = self.waypoints[0][0]
        odom.pose.pose.position.y = self.waypoints[0][1]
        received_waypoint = self.send_odom_and_read_next_waypoint(odom=odom)
        self.assertTrue(self.compare_vectors(received_waypoint, self.waypoints[1]))

    def test_waypoint_indicator(self):
        self.start_test()
        self.waypoints = rospy.get_param('/world/waypoints')
        print(f'WAYPOINTS: {self.waypoints}')
        stime = time.time()
        max_duration = 100
        while time.time() < stime + max_duration \
                and '/waypoint_indicator/current_waypoint' not in self.ros_topic.topic_values.keys():
            time.sleep(0.1)
        self._test_first_waypoint()
        self._test_transition_of_waypoint()

    def tearDown(self) -> None:
        self._ros_process.terminate()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
