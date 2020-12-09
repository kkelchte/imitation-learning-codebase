import os
import shutil
import time
import unittest

import rospy
from geometry_msgs.msg import Twist

from src.core.utils import get_filename_without_extension, get_data_dir
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.src.utils import euler_from_quaternion
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
            'gazebo': False,
            'fsm': False,
            'control_mapping': False,
            'output_path': self.output_dir,
            'modified_state_publisher': True,
            'modified_state_publisher_config': 'default',
        }

        # spinoff roslaunch
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=config,
                                       visible=False)

        # subscribe to modified_state_topic
        self.modified_state_topic = rospy.get_param('/robot/modified_state_sensor/topic')
        subscribe_topics = [
            TopicConfig(topic_name=self.modified_state_topic,
                        msg_type=rospy.get_param('/robot/modified_state_sensor/type')),
        ]
        # create publishers for all topics upon which modified state publisher depends
        self.tracking_pose_topic = rospy.get_param('/robot/tracking_position_sensor/topic')
        self.fleeing_pose_topic = rospy.get_param('/robot/fleeing_position_sensor/topic')
        publish_topics = [
            TopicConfig(topic_name=self.tracking_pose_topic,
                        msg_type=rospy.get_param('/robot/tracking_position_sensor/type')),
            TopicConfig(topic_name=self.fleeing_pose_topic,
                        msg_type=rospy.get_param('/robot/fleeing_position_sensor/type'))
        ]
        self.ros_topic = TestPublisherSubscriber(
            subscribe_topics=subscribe_topics,
            publish_topics=publish_topics
        )

    def test_modified_state_publisher(self):
        self.start()
        time.sleep(5)

        y_pos = 3
        tracking_pose = get_fake_pose_stamped(1, 2, 3, 0, 0, -0.258819, 0.9659258)
        fleeing_pose = get_fake_pose_stamped(4, 5, 6)
        self.ros_topic.publishers[self.tracking_pose_topic].publish(tracking_pose)
        self.ros_topic.publishers[self.fleeing_pose_topic].publish(fleeing_pose)

        #   wait safely
        max_duration = 30
        stime = time.time()
        while self.modified_state_topic not in self.ros_topic.topic_values.keys() \
                and time.time() - stime < max_duration:
            time.sleep(0.1)
        self.assertTrue(time.time()-stime < max_duration)
        time.sleep(1)
        roll, pitch, yaw = euler_from_quaternion((tracking_pose.pose.orientation.x,
                                                  tracking_pose.pose.orientation.y,
                                                  tracking_pose.pose.orientation.z,
                                                  tracking_pose.pose.orientation.w))
        modified_state = self.ros_topic.topic_values[self.modified_state_topic]
        self.assertTrue(modified_state.tracking_x == tracking_pose.pose.position.x)
        self.assertTrue(modified_state.tracking_y == tracking_pose.pose.position.y)
        self.assertTrue(modified_state.tracking_z == tracking_pose.pose.position.z)
        self.assertTrue(modified_state.fleeing_x == fleeing_pose.pose.position.x)
        self.assertTrue(modified_state.fleeing_y == fleeing_pose.pose.position.y)
        self.assertTrue(modified_state.fleeing_z == fleeing_pose.pose.position.z)
        self.assertTrue(modified_state.tracking_roll == roll)
        self.assertTrue(modified_state.tracking_pitch == pitch)
        self.assertAlmostEqual(modified_state.tracking_yaw, yaw)

    def tearDown(self) -> None:
        self._ros_process.terminate()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
