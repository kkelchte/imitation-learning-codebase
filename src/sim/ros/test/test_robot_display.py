import copy
import os
import shutil
import time
import unittest

import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose, PointStamped, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty, Header
from std_srvs.srv import Empty as Emptyservice, EmptyRequest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

from src.core.utils import get_filename_without_extension, get_to_root_dir, get_data_dir, safe_wait_till_true
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.src.utils import euler_from_quaternion, transform, process_image
from src.sim.ros.test.common_utils import TopicConfig, TestPublisherSubscriber, get_fake_laser_scan


class TestRobotDisplay(unittest.TestCase):

    def test_waypoints_tracking_real_bebop_with_KF_with_keyboard(self):
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

        self._config = {
            'output_path': self.output_dir,
            'world_name': 'hexagon',
            'robot_name': 'bebop_real',
            'gazebo': False,
            'fsm': True,
            'fsm_mode': 'TakeOverRun',
            'control_mapping': True,
            'control_mapping_config': 'mathias_controller_keyboard',
            'waypoint_indicator': True,
            'altitude_control': False,
            'mathias_controller_with_KF': True,
            'starting_height': 1.,
            'keyboard': True,
            'robot_display': True, 
            'mathias_controller_config_file_path_with_extension':
                f'{os.environ["CODEDIR"]}/src/sim/ros/config/actor/mathias_controller_with_KF_real_bebop.yml',
        }

        # spinoff roslaunch
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=self._config,
                                       visible=True)

        # subscribe to command control
        self.visualisation_topic = '/actor/mathias_controller/visualisation'
        subscribe_topics = [
            TopicConfig(topic_name=rospy.get_param('/robot/position_sensor/topic'),
                        msg_type=rospy.get_param('/robot/position_sensor/type')),
            TopicConfig(topic_name='/fsm/state',
                        msg_type='String'),
            TopicConfig(topic_name='/waypoint_indicator/current_waypoint', msg_type='Float32MultiArray'),
            TopicConfig(topic_name=self.visualisation_topic,
                        msg_type='Image')
        ]
        publish_topics = [
            TopicConfig(topic_name='/fsm/reset', msg_type='Empty'),
        ]

        self.ros_topic = TestPublisherSubscriber(
            subscribe_topics=subscribe_topics,
            publish_topics=publish_topics
        )

        safe_wait_till_true('"/fsm/state" in kwargs["ros_topic"].topic_values.keys()',
                            True, 25, 0.1, ros_topic=self.ros_topic)
        self.assertEqual(self.ros_topic.topic_values['/fsm/state'].data, FsmState.Unknown.name)

        while True:
            # publish reset
            self.ros_topic.publishers['/fsm/reset'].publish(Empty())

            while self.ros_topic.topic_values["/fsm/state"].data != FsmState.Running.name:
                rospy.sleep(0.1)
            safe_wait_till_true('"/waypoint_indicator/current_waypoint" in kwargs["ros_topic"].topic_values.keys()',
                                True, 10, 0.1, ros_topic=self.ros_topic)
            poses = []
            waypoints = []
            while self.ros_topic.topic_values["/fsm/state"].data != FsmState.Terminated.name and \
                    self.ros_topic.topic_values["/fsm/state"].data != FsmState.TakenOver.name:
                rospy.sleep(0.5)
                pose = self.get_pose()
                waypoint = self.ros_topic.topic_values['/waypoint_indicator/current_waypoint']
                poses.append(pose)
                waypoints.append(waypoint)

            plt.figure(figsize=(15, 15))
            plt.scatter([p[0] for p in poses],
                        [p[1] for p in poses],
                        color='C0', label='xy-pose')
            plt.scatter([p.data[0] for p in waypoints],
                        [p.data[1] for p in waypoints],
                        color='C1', label='xy-waypoints')
            plt.legend()
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.show()

    def tearDown(self) -> None:
        print(f'shutting down...')
        self._ros_process.terminate()
        # shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
