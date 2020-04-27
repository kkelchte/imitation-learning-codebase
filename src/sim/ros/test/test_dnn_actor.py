import os
import shutil
import time
import unittest

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image
import numpy as np

from src.core.utils import get_filename_without_extension
from src.core.data_types import ProcessState
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.test.common_utils import TopicConfig, TestPublisherSubscriber, get_fake_image


class TestDnnActor(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        config = {
            'robot_name': 'turtlebot_sim',
            'world_name': 'test_waypoints',
            'gazebo': False,
            'fsm': False,
            'control_mapping': False,
            'ros_expert': False,
            'dnn_actor': True,
            'dnn_config_file_path_with_extension': f'src/sim/ros/config/actor/dnn_actor.yml',
            'output_path': self.output_dir,
        }

        # spinoff roslaunch
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=config,
                                       visible=False)

        # subscribe to command control
        specs = rospy.get_param('/actor/dnn_actor/specs')
        self.command_topic = specs['command_topic']
        self._odom_topic = rospy.get_param('/robot/odometry_topic')
        subscribe_topics = [
            TopicConfig(topic_name=self.command_topic, msg_type="Twist"),
            TopicConfig(topic_name=self._odom_topic, msg_type="Odometry"),
        ]
        # create publishers for all relevant sensors < sensor expert
        self._image_topic = rospy.get_param('/robot/forward_camera_topic')
        self._image_type = rospy.get_param('/robot/forward_camera_type')

        publish_topics = [
            TopicConfig(topic_name=self._image_topic, msg_type=self._image_type),
        ]

        self.ros_topic = TestPublisherSubscriber(
            subscribe_topics=subscribe_topics,
            publish_topics=publish_topics
        )
        time.sleep(3)  # give model time to be uploaded.

    def send_image_and_read_twist(self, image: Image) -> Twist:
        self.ros_topic.publishers[self._image_topic].publish(image)
        time.sleep(3)
        received_twist: Twist = self.ros_topic.topic_values[self.command_topic]
        return received_twist

    def test_outputs_for_fake_forward_images(self):
        # publish depth scan making robot go left
        image = get_fake_image()
        twist = self.send_image_and_read_twist(image)
        self.assertTrue(twist.angular.z != 0)

    def tearDown(self) -> None:
        self._ros_process.terminate()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
