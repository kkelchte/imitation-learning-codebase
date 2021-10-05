import copy
import os
import shutil
import time
import unittest
import sys

import numpy as np
import rospy
from bebop_msgs.msg import CommonCommonStateBatteryStateChanged
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose, PointStamped, Point
from std_msgs.msg import Empty, Header, String
from std_srvs.srv import Empty as Emptyservice, EmptyRequest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

from src.core.utils import get_filename_without_extension, get_to_root_dir, get_data_dir, safe_wait_till_true
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.src.utils import euler_from_quaternion, transform, process_image
from src.sim.ros.test.common_utils import TopicConfig, TestPublisherSubscriber, get_random_twist, get_random_image, get_random_reference_pose

class TestRobotDisplay(unittest.TestCase):

    def test_start_up_and_send_random_input(self):
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

        self._config = {
            'output_path': self.output_dir,
            'world_name': 'default',
            'robot_name': 'bebop_real',
            'gazebo': False,
            'fsm': False,
            'control_mapping': False,
            'waypoint_indicator': False,
            'altitude_control': False,
            'mathias_controller_with_KF': False,
            'keyboard': False,
            'robot_display': False, 
        }

        # spinoff roslaunch
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=self._config,
                                       visible=True)

        # subscribe to command control
        publish_topics = [
            TopicConfig(topic_name='/fsm/reset', msg_type='Empty'),
            TopicConfig(topic_name='/fsm/state', msg_type='String'),
            TopicConfig(topic_name='/bebop/cmd_vel', msg_type='Twist'),
            TopicConfig(topic_name='/bebop/image_raw', msg_type='Image'),
            TopicConfig(topic_name='/mask', msg_type='Image'),
            TopicConfig(topic_name='/reference_pose', msg_type='PointStamped'),
            TopicConfig(topic_name='/bebop/states/common/CommonState/BatteryStateChanged', msg_type='CommonCommonStateBatteryStateChanged')
        ]

        self.ros_topic = TestPublisherSubscriber(
            subscribe_topics=[],
            publish_topics=publish_topics
        )

        while not rospy.is_shutdown():
            # put something random on each topic every second
            # reset
            if np.random.binomial(1, 0.2) == 1:
                self.ros_topic.publishers['/fsm/reset'].publish(Empty())
            state = np.random.choice(['Unknown', 'Running', 'TakenOver', 'Terminated', 'DriveBack'])
            self.ros_topic.publishers['/fsm/state'].publish(String(state))
            battery = CommonCommonStateBatteryStateChanged()
            battery.percent = np.random.randint(0, 100, dtype=np.uint8)
            self.ros_topic.publishers['/bebop/states/common/CommonState/BatteryStateChanged'].publish(battery)
            self.ros_topic.publishers['/bebop/cmd_vel'].publish(get_random_twist())
            self.ros_topic.publishers['/bebop/image_raw'].publish(get_random_image((200,200,3)))
            self.ros_topic.publishers['/mask'].publish(get_random_image((200,200,1)))
#            self.ros_topic.publishers['/reference_pose'].publish(get_random_reference_pose())       
            self.ros_topic.publishers['/reference_pose'].publish(PointStamped(point=Point(x=2, y=0.5, z=0.5)))       
            rospy.sleep(0.1)

    def tearDown(self) -> None:
        print(f'shutting down...')
        self._ros_process.terminate()


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
