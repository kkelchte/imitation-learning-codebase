import os
import shutil
import time
import unittest

import rospy
from geometry_msgs.msg import Twist

from src.core.utils import get_filename_without_extension, get_data_dir
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.test.common_utils import TestPublisherSubscriber, TopicConfig, get_fake_pose_stamped

"""
Check if states are properly modified
"""


class TestModifiedStatePublisher(unittest.TestCase):

    def start(self) -> None:
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        config = {
            'robot_name': 'double_drone_sim',
            'fsm': False,
            'control_mapping': False,
            'output_path': self.output_dir,
            'waypoint_indicator': False,
            'modified_state_publisher': True,
            'modified_state_publisher_config': 'default',
        }

        # spinoff roslaunch
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=config,
                                       visible=True)

        # subscribe to modified_state_topic
        self.modified_state_topic = rospy.get_param('/robot/modified_state_topic')
        subscribe_topics = [
            TopicConfig(topic_name=self.modified_state_topic,
                        msg_type=rospy.get_param('/robot/modified_state_type')),
        ]
        # create publishers for all topics upon which modified state publisher depends
        self.tracking_pose_topic = rospy.get_param('/robot/tracking_tf_topic')
        self.fleeing_pose_topic = rospy.get_param('/robot/fleeing_tf_topic')
        publish_topics = [
            TopicConfig(topic_name=self.tracking_pose_topic,
                        msg_type=rospy.get_param('/robot/tracking_tf_type')),
            TopicConfig(topic_name=self.fleeing_pose_topic,
                        msg_type=rospy.get_param('/robot/fleeing_tf_type'))
        ]
        self.ros_topic = TestPublisherSubscriber(
            subscribe_topics=subscribe_topics,
            publish_topics=publish_topics
        )

    def test_modified_state_publisher(self):
        self.start()
        time.sleep(5)

        y_pos = 3
        tracking_pose = get_fake_pose_stamped(0, 0, 1)
        fleeing_pose = get_fake_pose_stamped(0, y_pos, 1)
        self.ros_topic.publishers[self.tracking_pose_topic].publish(tracking_pose)
        self.ros_topic.publishers[self.fleeing_pose_topic].publish(fleeing_pose)

        #   wait safely
        max_duration = 30
        stime = time.time()
        while self.modified_state_topic not in self.ros_topic.topic_values.keys() \
                and time.time() - stime < max_duration:
            time.sleep(0.1)
        self.assertTrue(time.time()-stime < max_duration)

        modified_state = self.ros_topic.topic_values[self.modified_state_topic]
        self.assertTrue(modified_state.data[1] == y_pos)

    def tearDown(self) -> None:
        self._ros_process.terminate()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
