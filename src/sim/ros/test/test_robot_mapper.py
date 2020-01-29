import shutil
import subprocess
import time
import unittest
import os
import shlex

import rospy
from nav_msgs.msg import Odometry

from src.sim.ros.catkin_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.src.process_wrappers import RosWrapper, ProcessState
from src.sim.ros.src.utils import adapt_vector_to_odometry
from src.sim.ros.test.common_utils import TopicConfig, TestPublisherSubscriber


class TestRobotMapper(unittest.TestCase):

    def start_test(self) -> None:
        self.output_dir = f'test_dir/test_{os.path.basename(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

        config = {
            'output_path': self.output_dir,
            'world_name': 'test_robot_mapper',
            'robot_name': 'turtlebot_sim',
            'gazebo': False,
            'fsm': False,
            'control_mapping': False,
            'ros_expert': False,
            'waypoint_indicator': False,
            'robot_mapper': True
        }

        # spinoff roslaunch
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=config,
                                       visible=True)

        # create publishers for all relevant sensors < sensor expert
        self._pose_topic = rospy.get_param('/robot/odometry_topic')
        self._pose_type = rospy.get_param('/robot/odometry_type')
        self._fsm_topic = rospy.get_param('/fsm/state_topic')
        publish_topics = [
            TopicConfig(topic_name=self._pose_topic, msg_type=self._pose_type),
            TopicConfig(topic_name=self._fsm_topic, msg_type='String')
        ]

        self.ros_topic = TestPublisherSubscriber(
            subscribe_topics=[],
            publish_topics=publish_topics
        )

    def publish_odom(self, x: float = 0, y: float = 0, z: float = 0, yaw: float = 0):
        odom = adapt_vector_to_odometry((x, y, z, yaw))
        self.ros_topic.publishers[self._pose_topic].publish(odom)
        time.sleep(0.1)

    def test_waypoint_indicator(self):
        self.start_test()
        stime = time.time()
        max_duration = 100
        while time.time() < stime + max_duration \
                and not rospy.has_param('/output_path'):
            time.sleep(0.1)
        self.publish_odom(x=0, y=0, z=0, yaw=0)
        self.publish_odom(x=6, y=0, z=0, yaw=0)
        self.publish_odom(x=3, y=0, z=0, yaw=0.7)
        self.publish_odom(x=6, y=3, z=0, yaw=-0.7)
        self.publish_odom(x=3, y=0, z=0, yaw=-0.7+1.57)
        self.publish_odom(x=0, y=3, z=0, yaw=-0.7+1.57)
        self.ros_topic.publishers[self._fsm_topic].publish(FsmState.Terminated.name)
        self.assertTrue(os.path.isfile(os.path.join(self.output_dir, 'trajectory.png')))

    def tearDown(self) -> None:
        self.assertEqual(self._ros_process.terminate(),
                         ProcessState.Terminated)
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
