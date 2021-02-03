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
from scipy.spatial.transform import Rotation as R

from src.core.utils import get_filename_without_extension, get_to_root_dir, get_data_dir, safe_wait_till_true
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.src.utils import euler_from_quaternion, transform
from src.sim.ros.test.common_utils import TopicConfig, TestPublisherSubscriber, get_fake_laser_scan


class TestMathiasController(unittest.TestCase):

    @unittest.skip
    def test_node_without_gazebo(self):
        # Initialize settings
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

        self._config = {
            'output_path': self.output_dir,
            'world_name': 'empty',
            'robot_name': 'drone_sim',
            'control_mapping': False,
            'control_mapping_config': 'mathias_controller',
            'mathias_controller': True,
        }
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=self._config,
                                       visible=True)

        self.command_topic = '/actor/mathias_controller/cmd_vel'
        subscribe_topics = [
            TopicConfig(topic_name=self.command_topic, msg_type="Twist"),
        ]
        self._pose_topic = rospy.get_param('/robot/position_sensor/topic')
        self._pose_type = rospy.get_param('/robot/position_sensor/type')
        self._reference_topic = '/reference_pose'
        self._reference_type = 'PointStamped'
        publish_topics = [
            TopicConfig(topic_name=self._pose_topic, msg_type=self._pose_type),
            TopicConfig(topic_name=self._reference_topic, msg_type=self._reference_type)
        ]
        self.ros_topic = TestPublisherSubscriber(
            subscribe_topics=subscribe_topics,
            publish_topics=publish_topics
        )

        safe_wait_till_true('"/actor/mathias_controller/cmd_vel" in kwargs["ros_topic"].topic_values.keys()',
                            True, 10, 0.1, ros_topic=self.ros_topic)
        # Test X
        # place robot on 1 and reference 2, see that linear-ctr is positive, rest is small
        msg = Odometry()
        msg.pose.pose.position.x = 1
        msg.pose.pose.orientation.w = 1
        self.ros_topic.publishers[self._pose_topic].publish(msg)
        msg = PointStamped()
        msg.point.x = 2
        self.ros_topic.publishers[self._reference_topic].publish(msg)
        time.sleep(1)
        result = self.ros_topic.topic_values[self.command_topic]
        self.assertTrue(result.linear.x > 0)
        self.assertAlmostEqual(result.linear.y, 0)
        self.assertAlmostEqual(result.linear.z, 0)
        self.assertAlmostEqual(result.angular.x, 0)
        self.assertAlmostEqual(result.angular.y, 0)
        self.assertAlmostEqual(result.angular.z, 0)

        # place robot on 2 and reference 1, see that linear-ctr is negative, rest is small
        msg = Odometry()
        msg.pose.pose.position.x = 2
        msg.pose.pose.orientation.w = 1
        self.ros_topic.publishers[self._pose_topic].publish(msg)
        msg = PointStamped()
        msg.point.x = 1
        self.ros_topic.publishers[self._reference_topic].publish(msg)
        time.sleep(1)
        result = self.ros_topic.topic_values[self.command_topic]
        self.assertTrue(result.linear.x < 0)
        self.assertAlmostEqual(result.linear.y, 0)
        self.assertAlmostEqual(result.linear.z, 0)
        self.assertAlmostEqual(result.angular.x, 0)
        self.assertAlmostEqual(result.angular.y, 0)
        self.assertAlmostEqual(result.angular.z, 0)

        # Test Y
        # place robot on 1 and reference 2, see that linear-ctr is positive, rest is small
        msg = Odometry()
        msg.pose.pose.position.y = 1
        msg.pose.pose.orientation.w = 1
        self.ros_topic.publishers[self._pose_topic].publish(msg)
        msg = PointStamped()
        msg.point.y = 2
        self.ros_topic.publishers[self._reference_topic].publish(msg)
        time.sleep(1)
        result = self.ros_topic.topic_values[self.command_topic]
        self.assertTrue(result.linear.y > 0)

        # place robot on 2 and reference 1, see that linear-ctr is negative, rest is small
        msg = Odometry()
        msg.pose.pose.position.y = 2
        msg.pose.pose.orientation.w = 1
        self.ros_topic.publishers[self._pose_topic].publish(msg)
        msg = PointStamped()
        msg.point.y = 1
        self.ros_topic.publishers[self._reference_topic].publish(msg)
        time.sleep(1)
        result = self.ros_topic.topic_values[self.command_topic]
        self.assertTrue(result.linear.y < 0)

        # Test Z
        # place robot on 1 and reference 2, see that linear-ctr is positive, rest is small
        msg = Odometry()
        msg.pose.pose.position.z = 1
        msg.pose.pose.orientation.w = 1
        self.ros_topic.publishers[self._pose_topic].publish(msg)
        msg = PointStamped()
        msg.point.z = 2
        self.ros_topic.publishers[self._reference_topic].publish(msg)
        time.sleep(1)
        result = self.ros_topic.topic_values[self.command_topic]
        self.assertTrue(result.linear.z > 0)

        # place robot on 2 and reference 1, see that linear-ctr is negative, rest is small
        msg = Odometry()
        msg.pose.pose.position.z = 2
        msg.pose.pose.orientation.w = 1
        self.ros_topic.publishers[self._pose_topic].publish(msg)
        msg = PointStamped()
        msg.point.z = 1
        self.ros_topic.publishers[self._reference_topic].publish(msg)
        time.sleep(1)
        result = self.ros_topic.topic_values[self.command_topic]
        self.assertTrue(result.linear.z < 0)

    @unittest.skip
    def test_drone_world_positioning_in_gazebo(self) -> None:
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

        self._config = {
            'output_path': self.output_dir,
            'world_name': 'empty',
            'robot_name': 'drone_sim',
            'gazebo': True,
            'fsm': True,
            'fsm_mode': 'TakeOverRun',
            'control_mapping': True,
            'control_mapping_config': 'mathias_controller',
            'altitude_control': True,
            'mathias_controller': True,
            'starting_height': 1.
        }

        # spinoff roslaunch
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=self._config,
                                       visible=True)

        # subscribe to command control
        subscribe_topics = [
            TopicConfig(topic_name=rospy.get_param('/robot/position_sensor/topic'),
                        msg_type=rospy.get_param('/robot/position_sensor/type')),
            TopicConfig(topic_name='/fsm/state',
                        msg_type='String')
        ]
        self._reference_topic = '/reference_pose'
        self._reference_type = 'PointStamped'
        publish_topics = [
            TopicConfig(topic_name='/fsm/reset', msg_type='Empty'),
            TopicConfig(topic_name=self._reference_topic, msg_type=self._reference_type),
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

        # pause again before start
        self._pause_client.wait_for_service()
        self._pause_client.call()

        self._set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        measured_data = {}
        index = 0
        while True:
            # set gazebo model state
            model_state = ModelState()
            model_state.model_name = 'quadrotor'
            model_state.pose = Pose()
            self._set_model_state.wait_for_service()
            self._set_model_state(model_state)

            # publish reset
            self.ros_topic.publishers['/fsm/reset'].publish(Empty())

            self._unpause_client.wait_for_service()
            self._unpause_client.call()

            # gets fsm in taken over state
            safe_wait_till_true('kwargs["ros_topic"].topic_values["/fsm/state"].data',
                                FsmState.TakenOver.name, 2, 0.1, ros_topic=self.ros_topic)

            # altitude control brings drone to starting_height
            safe_wait_till_true('kwargs["ros_topic"].topic_values["/fsm/state"].data',
                                FsmState.Running.name, 20, 0.1, ros_topic=self.ros_topic)
            # tweak z, x, y parameters
            #index = self.tweak_separate_axis(index, measured_data)

            # tweak joint fly to point
            index = self.tweak_joint_axis(index, measured_data)

    def tweak_separate_axis(self, index, measured_data):
        # send out reference pose Z
        self._pause_client.wait_for_service()
        self._pause_client.call()

        self.ros_topic.publishers[self._reference_topic].publish(PointStamped(point=Point(z=3.)))

        self._unpause_client.wait_for_service()
        self._unpause_client.call()

        points = []
        for _ in range(100):
            rospy.sleep(0.1)
            odom = self.ros_topic.topic_values[rospy.get_param('/robot/position_sensor/topic')]
            points.append(odom.pose.pose.position.z - 3.0)
        plt.plot(points, 'r-', label='z')
        measured_data[index] = {'z': points}

        # send out reference pose X
        self._pause_client.wait_for_service()
        self._pause_client.call()

        self.ros_topic.publishers[self._reference_topic].publish(PointStamped(point=Point(x=2., z=3.)))

        self._unpause_client.wait_for_service()
        self._unpause_client.call()
        points = []

        for _ in range(100):
            rospy.sleep(0.1)
            odom = self.ros_topic.topic_values[rospy.get_param('/robot/position_sensor/topic')]
            points.append(odom.pose.pose.position.x - 2.)
        plt.plot(points, 'b', label='x')
        measured_data[index]['x'] = points

        # send out reference pose Y
        self._pause_client.wait_for_service()
        self._pause_client.call()

        self.ros_topic.publishers[self._reference_topic].publish(PointStamped(point=Point(x=2., y=2., z=3.)))

        self._unpause_client.wait_for_service()
        self._unpause_client.call()

        points = []
        for _ in range(100):
            rospy.sleep(0.1)
            odom = self.ros_topic.topic_values[rospy.get_param('/robot/position_sensor/topic')]
            points.append(odom.pose.pose.position.y - 2.)
        plt.plot(points, 'g', label='y')
        measured_data[index]['y'] = points

        self._pause_client.wait_for_service()
        self._pause_client.call()

        plt.legend()
        plt.show()

        colors = ['C0', 'C1', 'C2', 'C3', 'C4']
        for key in measured_data.keys():
            for a in measured_data[key].keys():
                if a == 'x':
                    style = '-'
                elif a == 'y':
                    style = '--'
                else:
                    style = ':'
                plt.plot(measured_data[key][a], linestyle=style,
                         color=colors[key % len(colors)], label=f'{key}: {a}')
        plt.legend()
        plt.show()
        index += 1
        index %= len(colors)
        return index

    def tweak_joint_axis(self, index, measured_data):
        # send out reference pose
        self._pause_client.wait_for_service()
        self._pause_client.call()

        self.ros_topic.publishers[self._reference_topic].publish(PointStamped(point=Point(x=2., y=2., z=3.)))

        self._unpause_client.wait_for_service()
        self._unpause_client.call()

        measured_data[index] = {'x': [],
                                'y': [],
                                'z': []}
        for _ in range(100):
            rospy.sleep(0.1)
            odom = self.ros_topic.topic_values[rospy.get_param('/robot/position_sensor/topic')]
            measured_data[index]['x'].append(odom.pose.pose.position.x - 2.)
            measured_data[index]['y'].append(odom.pose.pose.position.y - 2.)
            measured_data[index]['z'].append(odom.pose.pose.position.z - 3.)

        self._pause_client.wait_for_service()
        self._pause_client.call()

        colors = ['C0', 'C1', 'C2', 'C3', 'C4']
        for key in measured_data.keys():
            for a in measured_data[key].keys():
                if a == 'x':
                    style = '-'
                elif a == 'y':
                    style = '--'
                else:
                    style = ':'
                plt.plot(measured_data[key][a], linestyle=style, linewidth=3 if key == index else 1,
                         color=colors[key % len(colors)], label=f'{key}: {a}')
        plt.legend()
        plt.show()
        index += 1
        index %= len(colors)
        return index

    @unittest.skip
    def test_waypoints_tracking_in_gazebo(self):
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

        self._config = {
            'output_path': self.output_dir,
            'world_name': 'debug_drone',
            'robot_name': 'drone_sim',
            'gazebo': True,
            'fsm': True,
            'fsm_mode': 'TakeOverRun',
            'control_mapping': True,
            'control_mapping_config': 'mathias_controller',
            'waypoint_indicator': True,
            'altitude_control': True,
            'mathias_controller': True,
            'starting_height': 1.
        }

        # spinoff roslaunch
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=self._config,
                                       visible=True)

        # subscribe to command control
        subscribe_topics = [
            TopicConfig(topic_name=rospy.get_param('/robot/position_sensor/topic'),
                        msg_type=rospy.get_param('/robot/position_sensor/type')),
            TopicConfig(topic_name='/fsm/state',
                        msg_type='String'),
            TopicConfig(topic_name='/waypoint_indicator/current_waypoint', msg_type='Float32MultiArray')
        ]
        publish_topics = [
            TopicConfig(topic_name='/fsm/reset', msg_type='Empty'),
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

        # pause again before start
        self._pause_client.wait_for_service()
        self._pause_client.call()
        self._set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        measured_data = {}
        index = 0
        while True:
            measured_data[index] = {'x': [],
                                    'y': [],
                                    'z': []}
            # set gazebo model state
            model_state = ModelState()
            model_state.model_name = 'quadrotor'
            model_state.pose = Pose()
            self._set_model_state.wait_for_service()
            self._set_model_state(model_state)

            # publish reset
            self.ros_topic.publishers['/fsm/reset'].publish(Empty())

            self._unpause_client.wait_for_service()
            self._unpause_client.call()

            # gets fsm in taken over state
            safe_wait_till_true('kwargs["ros_topic"].topic_values["/fsm/state"].data',
                                FsmState.TakenOver.name, 2, 0.1, ros_topic=self.ros_topic)

            # altitude control brings drone to starting_height
            safe_wait_till_true('kwargs["ros_topic"].topic_values["/fsm/state"].data',
                                FsmState.Running.name, 20, 0.1, ros_topic=self.ros_topic)

            while self.ros_topic.topic_values["/fsm/state"].data != FsmState.Terminated.name:
                rospy.sleep(0.1)
                odom = self.ros_topic.topic_values[rospy.get_param('/robot/position_sensor/topic')]
                waypoint = self.ros_topic.topic_values['/waypoint_indicator/current_waypoint']
                measured_data[index]['x'].append(odom.pose.pose.position.x - waypoint.data[0])
                measured_data[index]['y'].append(odom.pose.pose.position.y - waypoint.data[1])
                measured_data[index]['z'].append(odom.pose.pose.position.z - 1)
            # see it reaches the goal state:
            safe_wait_till_true('kwargs["ros_topic"].topic_values["/fsm/state"].data',
                                FsmState.Terminated.name, 20, 0.1, ros_topic=self.ros_topic)

            self._pause_client.wait_for_service()
            self._pause_client.call()

            colors = ['C0', 'C1', 'C2', 'C3', 'C4']
            for key in measured_data.keys():
                for a in measured_data[key].keys():
                    if a == 'x':
                        style = '-'
                    elif a == 'y':
                        style = '--'
                    else:
                        style = ':'
                    plt.plot(measured_data[key][a], linestyle=style,
                             color=colors[key % len(colors)], label=f'{key}: {a}')
            plt.legend()
            plt.show()
            index += 1
            index %= len(colors)

    @unittest.skip
    def test_relative_reference_points_in_gazebo(self):
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

        self._config = {
            'output_path': self.output_dir,
            'world_name': 'empty',
            'robot_name': 'drone_sim',
            'gazebo': True,
            'fsm': True,
            'fsm_mode': 'TakeOverRun',
            'control_mapping': True,
            'control_mapping_config': 'mathias_controller',
            'altitude_control': True,
            'mathias_controller': True,
            'starting_height': 1.,
        }

        # spinoff roslaunch
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=self._config,
                                       visible=True)

        # subscribe to command control
        subscribe_topics = [
            TopicConfig(topic_name=rospy.get_param('/robot/position_sensor/topic'),
                        msg_type=rospy.get_param('/robot/position_sensor/type')),
            TopicConfig(topic_name='/fsm/state',
                        msg_type='String')
        ]
        self._reference_topic = '/reference_pose'
        self._reference_type = 'PointStamped'
        publish_topics = [
            TopicConfig(topic_name='/fsm/reset', msg_type='Empty'),
            TopicConfig(topic_name=self._reference_topic, msg_type=self._reference_type),
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

        # pause again before start
        self._pause_client.wait_for_service()
        self._pause_client.call()

        self._set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        measured_data = {}
        index = 0
        while True:
            # set gazebo model state
            model_state = ModelState()
            model_state.model_name = 'quadrotor'
            model_state.pose = Pose()
            self._set_model_state.wait_for_service()
            self._set_model_state(model_state)

            # publish reset
            self.ros_topic.publishers['/fsm/reset'].publish(Empty())

            self._unpause_client.wait_for_service()
            self._unpause_client.call()

            # gets fsm in taken over state
            safe_wait_till_true('kwargs["ros_topic"].topic_values["/fsm/state"].data',
                                FsmState.TakenOver.name, 2, 0.1, ros_topic=self.ros_topic)

            # altitude control brings drone to starting_height
            safe_wait_till_true('kwargs["ros_topic"].topic_values["/fsm/state"].data',
                                FsmState.Running.name, 20, 0.1, ros_topic=self.ros_topic)

            # tweak joint fly to point as relative pose
            # send out reference pose
            self._pause_client.wait_for_service()
            self._pause_client.call()

            self.ros_topic.publishers[self._reference_topic].publish(PointStamped(header=Header(frame_id="agent"),
                                                                                  point=Point(x=2., y=2., z=2.)))

            self._unpause_client.wait_for_service()
            self._unpause_client.call()

            measured_data[index] = {'x': [],
                                    'y': [],
                                    'z': []}
            for _ in range(100):
                rospy.sleep(0.1)
                odom = self.ros_topic.topic_values[rospy.get_param('/robot/position_sensor/topic')]
                measured_data[index]['x'].append(odom.pose.pose.position.x - 2.)
                measured_data[index]['y'].append(odom.pose.pose.position.y - 2.)
                measured_data[index]['z'].append(odom.pose.pose.position.z - 3.)

            self._pause_client.wait_for_service()
            self._pause_client.call()

            colors = ['C0', 'C1', 'C2', 'C3', 'C4']
            for key in measured_data.keys():
                for a in measured_data[key].keys():
                    if a == 'x':
                        style = '-'
                    elif a == 'y':
                        style = '--'
                    else:
                        style = ':'
                    plt.plot(measured_data[key][a], linestyle=style, linewidth=3 if key == index else 1,
                             color=colors[key % len(colors)], label=f'{key}: {a}')
            plt.legend()
            plt.show()
            index += 1
            index %= len(colors)

    @unittest.skip
    def test_drone_keyboard_gazebo(self):
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

        self._config = {
            'output_path': self.output_dir,
            'world_name': 'empty',
            'robot_name': 'drone_sim',
            'gazebo': True,
            'fsm': True,
            'fsm_mode': 'TakeOverRun',
            'control_mapping': True,
            'control_mapping_config': 'mathias_controller_keyboard',
            'altitude_control': False,
            'keyboard': True,
            'mathias_controller': True,
            'yaw_or': 1.5,
            'x_pos': 1,
            'y_pos': 2
        }

        # spinoff roslaunch
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=self._config,
                                       visible=True)

        # subscribe to command control
        subscribe_topics = [
            TopicConfig(topic_name=rospy.get_param('/robot/position_sensor/topic'),
                        msg_type=rospy.get_param('/robot/position_sensor/type')),
            TopicConfig(topic_name='/fsm/state',
                        msg_type='String')

        ]
        self._reference_topic = '/reference_pose'
        self._reference_type = 'PointStamped'
        publish_topics = [
            TopicConfig(topic_name='/fsm/reset', msg_type='Empty'),
            TopicConfig(topic_name=self._reference_topic, msg_type=self._reference_type),
        ]

        self.ros_topic = TestPublisherSubscriber(
            subscribe_topics=subscribe_topics,
            publish_topics=publish_topics
        )

        self._unpause_client = rospy.ServiceProxy('/gazebo/unpause_physics', Emptyservice)
        self._pause_client = rospy.ServiceProxy('/gazebo/pause_physics', Emptyservice)

        safe_wait_till_true('"/fsm/state" in kwargs["ros_topic"].topic_values.keys()',
                            True, 10, 0.1, ros_topic=self.ros_topic)
        measured_data = {}
        index = 0
        while True:
            # publish reset
            self.ros_topic.publishers['/fsm/reset'].publish(Empty())

            self._unpause_client.wait_for_service()
            self._unpause_client.call()

            #index = self.tweak_steady_pose(measured_data, index)
            #index = self.tweak_separate_axis_keyboard(measured_data, index, axis=0)
            index = self.tweak_combined_axis_keyboard(measured_data, index, point=[2, 2, 1])

    #@unittest.skip
    def test_drone_relative_positioning_real_bebop(self):
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

        self._config = {
            'output_path': self.output_dir,
            'world_name': 'empty',
            'robot_name': 'bebop_real',
            'gazebo': False,
            'fsm': True,
            'fsm_mode': 'TakeOverRun',
            'control_mapping': True,
            'control_mapping_config': 'mathias_controller_keyboard',
            'altitude_control': False,
            'keyboard': True,
            'mathias_controller': True,
        }

        # spinoff roslaunch
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=self._config,
                                       visible=True)

        # subscribe to command control
        subscribe_topics = [
            TopicConfig(topic_name=rospy.get_param('/robot/position_sensor/topic'),
                        msg_type=rospy.get_param('/robot/position_sensor/type')),
            TopicConfig(topic_name='/fsm/state',
                        msg_type='String')

        ]
        self._reference_topic = '/reference_pose'
        self._reference_type = 'PointStamped'
        publish_topics = [
            TopicConfig(topic_name='/fsm/reset', msg_type='Empty'),
            TopicConfig(topic_name=self._reference_topic, msg_type=self._reference_type),
        ]

        self.ros_topic = TestPublisherSubscriber(
            subscribe_topics=subscribe_topics,
            publish_topics=publish_topics
        )
        safe_wait_till_true('"/fsm/state" in kwargs["ros_topic"].topic_values.keys()',
                            True, 10, 0.1, ros_topic=self.ros_topic)
        measured_data = {}
        index = 0
        while True:
            # publish reset
            self.ros_topic.publishers['/fsm/reset'].publish(Empty())
            #index = self.tweak_steady_pose(measured_data, index)
            index = self.tweak_separate_axis_keyboard(measured_data, index, axis=0)

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

    def tweak_steady_pose(self, measured_data, index):
        # gets fsm in taken over state
        safe_wait_till_true('kwargs["ros_topic"].topic_values["/fsm/state"].data',
                            FsmState.TakenOver.name, 20, 0.1, ros_topic=self.ros_topic)

        # once drone is in good starting position invoke 'go' with key m
        while self.ros_topic.topic_values["/fsm/state"].data != FsmState.Running.name:
            # with absolute positioning
            ref_pose = self.get_pose()
            if ref_pose is not None:
                self.ros_topic.publishers[self._reference_topic].publish(PointStamped(header=Header(),
                                                                                      point=Point(x=ref_pose[0],
                                                                                                  y=ref_pose[1],
                                                                                                  z=ref_pose[2])))
            # with relative positioning
            #self.ros_topic.publishers[self._reference_topic].publish(PointStamped(header=Header(frame_id="agent"),
            #                                                                      point=Point(x=0., y=0., z=0.)))
            rospy.sleep(0.5)

        measured_data[index] = {'x': [],
                                'y': [],
                                'z': [],
                                'yaw': []}

        # Mathias controller should keep drone in steady pose
        while self.ros_topic.topic_values["/fsm/state"].data != FsmState.TakenOver.name:
            x, y, z, yaw = self.get_pose()
            measured_data[index]['x'].append(x - ref_pose[0])
            measured_data[index]['y'].append(y - ref_pose[1])
            measured_data[index]['z'].append(z - ref_pose[2])
            measured_data[index]['yaw'].append(yaw - ref_pose[3])
            rospy.sleep(0.5)

        colors = ['C0', 'C1', 'C2', 'C3', 'C4']
        styles = {'x': '-', 'y': '--', 'z': ':', 'yaw': '-.'}
        for key in measured_data.keys():
            for a in measured_data[key].keys():
                plt.plot(measured_data[key][a], linestyle=styles[a], linewidth=3 if key == index else 1,
                         color=colors[key % len(colors)], label=f'{key}: {a}')
        plt.legend()
        plt.show()
        index += 1
        index %= len(colors)
        return index

    def tweak_separate_axis_keyboard(self, measured_data, index, axis=2):
        # gets fsm in taken over state
        safe_wait_till_true('kwargs["ros_topic"].topic_values["/fsm/state"].data',
                            FsmState.TakenOver.name, 20, 0.1, ros_topic=self.ros_topic)
        d = 1
        point = [d if axis == 0 else 0.,
                 d if axis == 1 else 0.,
                 d if axis == 2 else 0.]
        # gets fsm in taken over state
        safe_wait_till_true('kwargs["ros_topic"].topic_values["/fsm/state"].data',
                            FsmState.TakenOver.name, 20, 0.1, ros_topic=self.ros_topic)

        # once drone is in good starting position invoke 'go' with key m
        while self.ros_topic.topic_values["/fsm/state"].data != FsmState.Running.name:
            # while taking off, update reference point for PID controller to remain at same height
            self.ros_topic.publishers[self._reference_topic].publish(PointStamped(header=Header(frame_id="agent"),
                                                                                  point=Point(x=point[0],
                                                                                              y=point[1],
                                                                                              z=point[2])
                                                                                  ))
            last_pose = self.get_pose()
            rospy.sleep(0.5)

        measured_data[index] = {'x': [],
                                'y': [],
                                'z': [],
                                'yaw': []}
        goal_pose = transform(points=[np.asarray(point)],
                              orientation=R.from_euler('XYZ', (0, 0, last_pose[-1]), degrees=False).as_matrix(),
                              translation=np.asarray(last_pose[:3]))[0]

        # Mathias controller should keep drone in steady pose
        while self.ros_topic.topic_values["/fsm/state"].data != FsmState.TakenOver.name:
            pose = self.get_pose()
            pose_error = Point(
                x=pose[0] - goal_pose[0],
                y=pose[1] - goal_pose[1],
                z=pose[2] - goal_pose[2])
            # rotate pose error to global yaw frame
            pose_error = transform(points=[pose_error],
                                   orientation=R.from_euler('XYZ', (0, 0, -last_pose[0]),
                                                            degrees=False).as_matrix())[0]
            measured_data[index]['x'].append(pose_error.x)
            measured_data[index]['y'].append(pose_error.y)
            measured_data[index]['z'].append(pose_error.z)
            measured_data[index]['yaw'].append(last_pose[-1])
            rospy.sleep(0.5)

        colors = ['C0', 'C1', 'C2', 'C3', 'C4']
        styles = {'x': '-', 'y': '--', 'z': ':', 'yaw': '-.'}
        fig = plt.figure(figsize=(15, 15))
        for key in measured_data.keys():
            for a in measured_data[key].keys():
                plt.plot(measured_data[key][a], linestyle=styles[a], linewidth=3 if key == index else 1,
                         color=colors[key % len(colors)], label=f'{key}: {a}')
        plt.legend()
        plt.show()
        index += 1
        index %= len(colors)
        return index

    def tweak_combined_axis_keyboard(self, measured_data, index, point):
        # gets fsm in taken over state
        safe_wait_till_true('kwargs["ros_topic"].topic_values["/fsm/state"].data',
                            FsmState.TakenOver.name, 20, 0.1, ros_topic=self.ros_topic)

        # once drone is in good starting position invoke 'go' with key m
        while self.ros_topic.topic_values["/fsm/state"].data != FsmState.Running.name:
            # while taking off, update reference point for PID controller to remain at same height
            self.ros_topic.publishers[self._reference_topic].publish(PointStamped(header=Header(frame_id="agent"),
                                                                                  point=Point(x=point[0],
                                                                                              y=point[1],
                                                                                              z=point[2])
                                                                                  ))
            last_pose = self.get_pose()
            rospy.sleep(0.5)

        measured_data[index] = {'x': [],
                                'y': [],
                                'z': [],
                                'yaw': []}
        goal_pose = transform(points=[np.asarray(point)],
                              orientation=R.from_euler('XYZ', (0, 0, last_pose[-1]), degrees=False).as_matrix(),
                              translation=np.asarray(last_pose[:3]))[0]

        # Mathias controller should keep drone in steady pose
        while self.ros_topic.topic_values["/fsm/state"].data != FsmState.TakenOver.name:
            pose = self.get_pose()
            pose_error = Point(
                x=pose[0] - goal_pose[0],
                y=pose[1] - goal_pose[1],
                z=pose[2] - goal_pose[2])
            # rotate pose error to global yaw frame
            pose_error = transform(points=[pose_error],
                                   orientation=R.from_euler('XYZ', (0, 0, -last_pose[0]),
                                                            degrees=False).as_matrix())[0]
            measured_data[index]['x'].append(pose_error.x)
            measured_data[index]['y'].append(pose_error.y)
            measured_data[index]['z'].append(pose_error.z)
            measured_data[index]['yaw'].append(last_pose[-1])
            rospy.sleep(0.5)

        colors = ['C0', 'C1', 'C2', 'C3', 'C4']
        styles = {'x': '-', 'y': '--', 'z': ':', 'yaw': '-.'}
        fig = plt.figure(figsize=(15, 15))
        for key in measured_data.keys():
            for a in measured_data[key].keys():
                plt.plot(measured_data[key][a], linestyle=styles[a], linewidth=3 if key == index else 1,
                         color=colors[key % len(colors)], label=f'{key}: {a}')
        plt.legend()
        plt.show()
        index += 1
        index %= len(colors)
        return index

    def tearDown(self) -> None:
        self._ros_process.terminate()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
