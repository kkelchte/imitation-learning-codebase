import os
import shutil
import time
import unittest

import numpy as np
import matplotlib.pyplot as plt
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose
from std_msgs.msg import Empty
from std_srvs.srv import Empty as Emptyservice, EmptyRequest

from src.core.utils import get_filename_without_extension, get_to_root_dir, get_data_dir, safe_wait_till_true
from src.core.data_types import TerminationType, SensorType
from src.sim.common.environment import EnvironmentConfig
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.src.ros_environment import RosEnvironment
from src.sim.ros.src.utils import quaternion_from_euler, euler_from_quaternion, process_image
from src.sim.ros.test.common_utils import TopicConfig, TestPublisherSubscriber


class TestChessboardWorld(unittest.TestCase):

    #@unittest.skip
    def test_forward_cam_drone(self) -> None:
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

        self._config = {
            'output_path': self.output_dir,
            'world_name': 'chessboard',
            'robot_name': 'drone_sim',
            'gazebo': True,
            'fsm': True,
            'fsm_mode': 'TakeOverRun',
            'control_mapping': True,
            'control_mapping_config': 'mathias_controller',
            'altitude_control': True,
            'waypoint_indicator': False,
            'chessboard_detector': True,
            'mathias_controller_with_KF': True,
            'mathias_controller_config_file_path_with_extension':
                f'{os.environ["CODEDIR"]}/src/sim/ros/config/actor/mathias_controller_with_KF.yml',
            'starting_height': 1,
            'x_pos': -2,
        }

        # spinoff roslaunch
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=self._config,
                                       visible=True)

        # subscribe to command control
        self.image_topic = rospy.get_param(f'/robot/camera_sensor/topic')
        subscribe_topics = [
            TopicConfig(topic_name=rospy.get_param('/robot/position_sensor/topic'),
                        msg_type=rospy.get_param('/robot/position_sensor/type')),
            TopicConfig(topic_name='/waypoint_indicator/current_waypoint',
                        msg_type='Float32MultiArray'),
            TopicConfig(topic_name='/fsm/state',
                        msg_type='String'),
            TopicConfig(topic_name=self.image_topic,
                        msg_type='Image')
        ]

        publish_topics = [
            TopicConfig(topic_name='/fsm/reset', msg_type='Empty')
        ]

        self.ros_topic = TestPublisherSubscriber(
            subscribe_topics=subscribe_topics,
            publish_topics=publish_topics
        )
        self._unpause_client = rospy.ServiceProxy('/gazebo/unpause_physics', Emptyservice)
        self._pause_client = rospy.ServiceProxy('/gazebo/pause_physics', Emptyservice)
        self._set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        # unpause gazebo to receive messages
        self._unpause_client.wait_for_service()
        self._unpause_client.call()

        safe_wait_till_true('"/fsm/state" in kwargs["ros_topic"].topic_values.keys()',
                            True, 10, 0.1, ros_topic=self.ros_topic)
        self.assertEqual(self.ros_topic.topic_values['/fsm/state'].data, FsmState.Unknown.name)

        # publish reset
        self.ros_topic.publishers['/fsm/reset'].publish(Empty())

        # gets fsm in taken over state
        safe_wait_till_true('kwargs["ros_topic"].topic_values["/fsm/state"].data',
                            FsmState.TakenOver.name, 2, 0.1, ros_topic=self.ros_topic)

        # altitude control brings drone to starting_height
        safe_wait_till_true('kwargs["ros_topic"].topic_values["/fsm/state"].data',
                            FsmState.Running.name, 45, 0.1, ros_topic=self.ros_topic)

        # check pose
        pose = self.get_pose()
        frame = process_image(self.ros_topic.topic_values[self.image_topic])
        plt.figure(figsize=(15, 20))
        plt.imshow(frame)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{os.environ["HOME"]}/chessboard_view_{pose[0]}_{pose[1]}_{pose[2]}_{pose[3]}.png')

        a = 100
        # TODO save image
        self._pause_client.wait_for_service()
        self._pause_client.call()

    def get_pose(self):
        if rospy.get_param('/robot/position_sensor/topic') in self.ros_topic.topic_values.keys():
            odom = self.ros_topic.topic_values[rospy.get_param('/robot/position_sensor/topic')]
            quaternion = (odom.pose.pose.orientation.x,
                          odom.pose.pose.orientation.y,
                          odom.pose.pose.orientation.z,
                          odom.pose.pose.orientation.w)
            _, _, yaw = euler_from_quaternion(quaternion)
            return odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z, yaw
        else:
            return None

    def tearDown(self) -> None:
        self._ros_process.terminate()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
