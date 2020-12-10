#!/usr/bin/python3.8
import os
import shutil
import unittest
import time

import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Twist, Pose
from hector_uav_msgs.srv import EnableMotors
from std_srvs.srv import Empty as Emptyservice, EmptyRequest

from src.core.utils import count_grep_name, get_data_dir, safe_wait_till_true
from src.core.utils import get_filename_without_extension
from src.core.data_types import ProcessState
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.src.utils import quaternion_from_euler
from src.sim.ros.test.common_utils import TestPublisherSubscriber, TopicConfig


class TestRos(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

    @unittest.skip
    def test_launch_and_terminate_ros(self):
        ros_process = RosWrapper(launch_file='empty_ros.launch',
                                 config={
                                     'output_path': self.output_dir
                                 })
        self.assertEqual(ros_process.get_state(), ProcessState.Running)
        self.assertTrue(count_grep_name('ros') > 0)
        ros_process.terminate()

    @unittest.skip
    def test_launch_and_terminate_gazebo(self):
        random_seed = 123
        config = {
            'random_seed': random_seed,
            'gazebo': 'true',
            'world_name': 'cube_world',
            'output_path': self.output_dir
        }
        ros_process = RosWrapper(launch_file='load_ros.launch',
                                 config=config,
                                 visible=False)
        self.assertEqual(ros_process.get_state(), ProcessState.Running)
        time.sleep(5)
        self.assertGreaterEqual(count_grep_name('gzserver'), 1)
        ros_process.terminate()
        self.assertEqual(count_grep_name('gzserver'), 0)

    @unittest.skip
    def test_load_params(self):
        config = {
            'robot_name': 'turtlebot_sim',
            'fsm': False,
            'output_path': self.output_dir
        }
        ros_process = RosWrapper(launch_file='load_ros.launch',
                                 config=config,
                                 visible=True)
        self.assertEqual(rospy.get_param('/robot/camera_sensor/topic'), '/wa/camera/image_raw')
        ros_process.terminate()

    @unittest.skip
    def test_full_config(self):
        config = {
          "output_path": self.output_dir,
          "random_seed": 123,
          "robot_name": "turtlebot_sim",
          "fsm_mode": "SingleRun",  # file with fsm params loaded from config/fsm
          "fsm": True,
          "control_mapping": True,
          "waypoint_indicator": True,
          "control_mapping_config": "debug",
          "world_name": "debug_turtle",
          "x_pos": 0.0,
          "y_pos": 0.0,
          "z_pos": 0.0,
          "yaw_or": 1.57,
          "gazebo": True,
        }
        ros_process = RosWrapper(launch_file='load_ros.launch',
                                 config=config,
                                 visible=False)
        ros_process.terminate()

    @unittest.skip
    def test_takeoff_drone_sim(self):
        config = {
          "output_path": self.output_dir,
          "random_seed": 123,
          "robot_name": "drone_sim",
          "world_name": "empty",
          "gazebo": True,
        }
        ros_process = RosWrapper(launch_file='load_ros.launch',
                                 config=config,
                                 visible=False)
        self._unpause_client = rospy.ServiceProxy('/gazebo/unpause_physics', Emptyservice)
        self._pause_client = rospy.ServiceProxy('/gazebo/pause_physics', Emptyservice)
        self.ros_topic = TestPublisherSubscriber(
            subscribe_topics=[TopicConfig(topic_name=rospy.get_param('/robot/position_sensor/topic'),
                                          msg_type=rospy.get_param('/robot/position_sensor/type'))],
            publish_topics=[TopicConfig(topic_name=rospy.get_param('/robot/command_topic'),
                                        msg_type='Twist')]
        )
        # unpause gazebo to receive messages
        self._unpause_client.wait_for_service()
        self._unpause_client.call()

        safe_wait_till_true(f'\'{rospy.get_param("/robot/position_sensor/topic")}\' '
                            f'in kwargs["ros_topic"].topic_values.keys()',
                            True, 5, 0.1, ros_topic=self.ros_topic)

        rospy.wait_for_service('/enable_motors')
        enable_motors_service = rospy.ServiceProxy('/enable_motors', EnableMotors)
        enable_motors_service.call(False)

        self._pause_client.wait_for_service()
        self._pause_client.call()

        self._set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        for _ in range(3):

            # set gazebo model state
            model_state = ModelState()
            model_state.model_name = 'quadrotor'
            model_state.pose = Pose()
            self._set_model_state.wait_for_service()
            self._set_model_state(model_state)

            self._unpause_client.wait_for_service()
            self._unpause_client.call()
            enable_motors_service.call(True)

            # fly up till reference height
            reference = 1.
            errors = [1]
            while np.mean(errors) > 0.05:
                position = self.ros_topic.topic_values[rospy.get_param('/robot/position_sensor/topic')].pose.pose.position
                print(position.z)
                errors.append(np.abs(reference - position.z))
                if len(errors) > 5:
                    errors.pop(0)
                twist = Twist()
                twist.linear.z = +0.5 if position.z < reference else -0.5
                self.ros_topic.publishers[rospy.get_param('/robot/command_topic')].publish(twist)
                time.sleep(0.1)
            final_error = abs(self.ros_topic.topic_values[rospy.get_param('/robot/position_sensor/topic')].pose.pose.position.z - reference)
            self.assertLess(final_error, 0.1)
            self._pause_client.wait_for_service()
            self._pause_client(EmptyRequest())
            enable_motors_service.call(False)
        ros_process.terminate()

    # @unittest.skip
    def test_takeoff_double_drone_sim(self):
        config = {
          "output_path": self.output_dir,
          "random_seed": 123,
          "robot_name": "double_drone_sim",
          "world_name": "empty",
          "gazebo": True,
        }
        ros_process = RosWrapper(launch_file='load_ros.launch',
                                 config=config,
                                 visible=False)
        self._unpause_client = rospy.ServiceProxy('/gazebo/unpause_physics', Emptyservice)
        self._pause_client = rospy.ServiceProxy('/gazebo/pause_physics', Emptyservice)
        self.ros_topic = TestPublisherSubscriber(
            subscribe_topics=[TopicConfig(topic_name=rospy.get_param('/robot/tracking_position_sensor/topic'),
                                          msg_type=rospy.get_param('/robot/tracking_position_sensor/type')),
                              TopicConfig(topic_name=rospy.get_param('/robot/fleeing_position_sensor/topic'),
                                          msg_type=rospy.get_param('/robot/fleeing_position_sensor/type'))],
            publish_topics=[TopicConfig(topic_name=rospy.get_param('/robot/tracking_command_topic'),
                                        msg_type='Twist'),
                            TopicConfig(topic_name=rospy.get_param('/robot/fleeing_command_topic'),
                                        msg_type='Twist')])
        # unpause gazebo to receive messages
        self._unpause_client.wait_for_service()
        self._unpause_client.call()

        safe_wait_till_true(f'\'{rospy.get_param("/robot/tracking_position_sensor/topic")}\' '
                            f'in kwargs["ros_topic"].topic_values.keys()',
                            True, 5, 0.1, ros_topic=self.ros_topic)

        rospy.wait_for_service('/tracking/enable_motors')
        enable_motors_service_tracking = rospy.ServiceProxy('/tracking/enable_motors', EnableMotors)
        rospy.wait_for_service('/fleeing/enable_motors')
        enable_motors_service_fleeing = rospy.ServiceProxy('/fleeing/enable_motors', EnableMotors)
        # set gazebo model state
        self._set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        def set_model_state(name: str):
            model_state = ModelState()
            model_state.pose = Pose()
            model_state.model_name = name
            model_state.pose.position.x = 0
            if 'fleeing' in name:
                model_state.pose.position.x += 3.  # fixed distance
            model_state.pose.position.y = 0
            model_state.pose.position.z = 0
            yaw = 0 if 'fleeing' not in name else 3.14
            model_state.pose.orientation.x, model_state.pose.orientation.y, model_state.pose.orientation.z, \
                model_state.pose.orientation.w = quaternion_from_euler((0, 0, yaw))
            self._set_model_state.wait_for_service()
            self._set_model_state(model_state)

        self._pause_client.wait_for_service()
        self._pause_client.call()

        for _ in range(3):
            for model_name in rospy.get_param('/robot/model_name'):
                set_model_state(model_name)
            self._unpause_client.wait_for_service()
            self._unpause_client.call()

            # start motors
            enable_motors_service_tracking.call(True)
            time.sleep(3)
            enable_motors_service_fleeing.call(True)

            # fly up till reference height
            reference = 1.
            errors = [1]
            while np.mean(errors) > 0.1:
                for agent in ['tracking', 'fleeing']:
                    position = self.ros_topic.topic_values[rospy.get_param(f'/robot/{agent}_position_sensor/topic')].pose.position
                    print(f'{agent}: {position.z}')
                    errors.append(np.abs(reference - position.z))
                    twist = Twist()
                    twist.linear.z = +0.5 if position.z < reference else -0.5
                    self.ros_topic.publishers[rospy.get_param(f'/robot/{agent}_command_topic')].publish(twist)
                while len(errors) > 10:
                    errors.pop(0)
                time.sleep(0.1)

            final_error = abs(self.ros_topic.topic_values[rospy.get_param('/robot/tracking_position_sensor/topic')].pose.position.z - reference)
            self.assertLess(final_error, 0.2)
            final_error = abs(self.ros_topic.topic_values[rospy.get_param('/robot/fleeing_position_sensor/topic')].pose.position.z - reference)
            self.assertLess(final_error, 0.2)
            self._pause_client.wait_for_service()
            self._pause_client(EmptyRequest())
            # enable_motors_service_tracking.call(False)
            # time.sleep(3)
            # enable_motors_service_fleeing.call(False)

        ros_process.terminate()

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
