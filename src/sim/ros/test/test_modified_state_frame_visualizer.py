import os
import shutil
import time
import unittest

import matplotlib.pyplot as plt

import rospy
from geometry_msgs.msg import Twist
from imitation_learning_ros_package.msg import CombinedGlobalPoses
from std_msgs.msg import Float32MultiArray

from src.core.utils import get_filename_without_extension, get_data_dir, safe_wait_till_true
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.src.utils import euler_from_quaternion, process_float32multi_array, process_image
from src.sim.ros.test.common_utils import TestPublisherSubscriber, TopicConfig, get_fake_pose_stamped, \
    get_fake_combined_global_poses

"""
Check if states are properly modified
"""


class TestModifiedStatePublisher(unittest.TestCase):

    def test_modified_state_frame_visualizer(self):
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        config = {
            'output_path': self.output_dir,
            'modified_state_publisher_mode': 'CombinedGlobalPoses',
            'modified_state_frame_visualizer': True,
        }

        # spinoff roslaunch
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=config,
                                       visible=False)

        # subscribe to modified_state_topic
        self.frame_topic = '/modified_state_frame'
        subscribe_topics = [
            TopicConfig(topic_name=self.frame_topic,
                        msg_type='Image'),
        ]
        # create publishers for modified state
        self.modified_state_topic = '/modified_state'
        publish_topics = [
            TopicConfig(topic_name=self.modified_state_topic,
                        msg_type='CombinedGlobalPoses'),
        ]
        self.ros_topic = TestPublisherSubscriber(
            subscribe_topics=subscribe_topics,
            publish_topics=publish_topics
        )
        # test center view
        message = get_fake_combined_global_poses(0, 0, 1, 1, 0, 1, 0, 0, 0)
        rospy.sleep(2)
        self.ros_topic.publishers[self.modified_state_topic].publish(message)

        safe_wait_till_true('"/modified_state_frame" in kwargs["ros_topic"].topic_values.keys()',
                            True, 3, 0.1, ros_topic=self.ros_topic)
        frame = process_image(self.ros_topic.topic_values['/modified_state_frame'])
        # TODO: self.assertEqual(frame.sum(), CORRECT SUM)
        # plt.imshow(frame)
        # plt.show()

    def tearDown(self) -> None:
        self._ros_process.terminate()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
