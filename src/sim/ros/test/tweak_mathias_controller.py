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


class TestMathiasController(unittest.TestCase):

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
        self.visualisation_topic = None
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
                            True, 30, 0.1, ros_topic=self.ros_topic)
        measured_data = {}
        index = 0
        while True:
            # publish reset
            self.ros_topic.publishers['/fsm/reset'].publish(Empty())

            self._unpause_client.wait_for_service()
            self._unpause_client.call()

            index = self.tweak_combined_axis_keyboard(measured_data, index, point_ref_drone=[1, 0, 0])
            index = self.tweak_combined_axis_keyboard(measured_data, index, point_ref_drone=[0, 1, 0])
            index = self.tweak_combined_axis_keyboard(measured_data, index, point_ref_drone=[2, 2, 0])

    @unittest.skip
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
            'mathias_controller_config_file_path_with_extension': f'{os.environ["CODEDIR"]}/src/sim/ros/config/actor/mathias_controller_real_bebop.yml'
        }

        # spinoff roslaunch
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=self._config,
                                       visible=True)
        self.visualisation_topic = None
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
            # index = self.tweak_steady_pose(measured_data, index)
            #index = self.tweak_separate_axis_keyboard(measured_data, index, axis=0)
            index = self.tweak_combined_axis_keyboard(measured_data, index, point_ref_drone=[0.5, 0.5, 0.5])

    def get_pose(self):
        """
        returns x, y, z, yaw of drone in global frame
        """
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

    def tweak_combined_axis_keyboard(self, measured_data, index, point_ref_drone):
        initial_point_ref_drone = copy.deepcopy(point_ref_drone)
        # gets fsm in taken over state
        safe_wait_till_true('kwargs["ros_topic"].topic_values["/fsm/state"].data',
                            FsmState.TakenOver.name, 20, 0.1, ros_topic=self.ros_topic)

        # once drone is in good starting position invoke 'go' with key m
        while self.ros_topic.topic_values["/fsm/state"].data != FsmState.Running.name:
            # while taking off, update reference point for PID controller to remain at same height
            self.ros_topic.publishers[self._reference_topic].publish(PointStamped(header=Header(frame_id="agent"),
                                                                                  point=Point(x=point_ref_drone[0],
                                                                                              y=point_ref_drone[1],
                                                                                              z=point_ref_drone[2])))
            last_pose = self.get_pose()
            rospy.sleep(0.5)

        measured_data[index] = {'x': [],
                                'y': [],
                                'z': [],
                                'yaw': []}
        point_ref_global = transform(points=[np.asarray(point_ref_drone)],
                                     orientation=R.from_euler('XYZ', (0, 0, last_pose[-1]), degrees=False).as_matrix(),
                                     translation=np.asarray(last_pose[:3]))[0]

        # Mathias controller should keep drone in steady pose
        while self.ros_topic.topic_values["/fsm/state"].data != FsmState.TakenOver.name:
            point_drone_global = self.get_pose()
            point_ref_drone = Point(
                x=point_ref_global[0] - point_drone_global[0],
                y=point_ref_global[1] - point_drone_global[1],
                z=point_ref_global[2] - point_drone_global[2])
            # rotate pose error to rotated frame
            pose_error_local = transform(points=[point_ref_drone],
                                         orientation=R.from_euler('XYZ', (0, 0, -last_pose[-1]),
                                                                  degrees=False).as_matrix())[0]
            measured_data[index]['x'].append(pose_error_local.x)
            measured_data[index]['y'].append(pose_error_local.y)
            measured_data[index]['z'].append(pose_error_local.z)
            # measured_data[index]['yaw'].append(last_pose[-1])
            rospy.sleep(0.5)

        if False and '/bebop/land' in self.ros_topic.publishers.keys():
            self.ros_topic.publishers['/bebop/land'].publish(Empty())

        colors = ['C0', 'C1', 'C2', 'C3', 'C4']
        styles = {'x': '-', 'y': '--', 'z': ':', 'yaw': '-.'}
        fig = plt.figure(figsize=(15, 15))
        for key in measured_data.keys():
            for a in measured_data[key].keys():
                if len(measured_data[key][a]) != 0:
                    plt.plot(measured_data[key][a], linestyle=styles[a], linewidth=3 if key == index else 1,
                         color=colors[key % len(colors)], label=f'{key}: {a}')
        plt.legend()
        #plt.savefig(os.path.join(self.output_dir, 'measured_data.jpg'))
        plt.show()

        # print visualisation if it's in ros topic:
        if self.visualisation_topic in self.ros_topic.topic_values.keys():
            frame = process_image(self.ros_topic.topic_values[self.visualisation_topic])
            plt.figure(figsize=(15, 20))
            plt.imshow(frame)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'{os.environ["HOME"]}/kf_study/pid_tweak_{initial_point_ref_drone[0]}_{initial_point_ref_drone[1]}_{initial_point_ref_drone[2]}.png')
        plt.clf()
        plt.close()

        index += 1
        index %= len(colors)
        return index

    @unittest.skip
    def test_drone_keyboard_gazebo_with_KF_waypoints(self):
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

        self._config = {
            'output_path': self.output_dir,
            'world_name': 'gate_test',
            'robot_name': 'drone_sim',
            'gazebo': True,
            'fsm': True,
            'fsm_mode': 'TakeOverRun',
            'waypoint_indicator': True,
            'control_mapping': True,
            'control_mapping_config': 'mathias_controller_keyboard',
            'altitude_control': False,
            'keyboard': True,
            'mathias_controller_with_KF': True,
            'mathias_controller_config_file_path_with_extension':
                f'{os.environ["CODEDIR"]}/src/sim/ros/config/actor/mathias_controller_with_KF.yml',
            'yaw_or': 0,
            'x_pos': 0,
            'y_pos': 0
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
            TopicConfig(topic_name=self.visualisation_topic,
                        msg_type='Image')
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
                            True, 30, 0.1, ros_topic=self.ros_topic)

        measured_data = {}
        index = 0
        while True:
            measured_data[index] = {'x': [],
                                    'y': [],
                                    'z': []}

            # publish reset
            self.ros_topic.publishers['/fsm/reset'].publish(Empty())

            self._unpause_client.wait_for_service()
            self._unpause_client.call()

            # User flies drone to starting position
            while self.ros_topic.topic_values["/fsm/state"].data == FsmState.TakenOver.name:
                time.sleep(1)

            while self.ros_topic.topic_values["/fsm/state"].data == FsmState.Running.name:
                rospy.sleep(0.1)
                odom = self.ros_topic.topic_values[rospy.get_param('/robot/position_sensor/topic')]
                waypoint = self.ros_topic.topic_values['/waypoint_indicator/current_waypoint']
                measured_data[index]['x'].append(odom.pose.pose.position.x - waypoint.data[0])
                measured_data[index]['y'].append(odom.pose.pose.position.y - waypoint.data[1])
                measured_data[index]['z'].append(odom.pose.pose.position.z - 1)
            # see it reaches the goal state:
            # safe_wait_till_true('kwargs["ros_topic"].topic_values["/fsm/state"].data',
            #                    FsmState.Terminated.name, 20, 0.1, ros_topic=self.ros_topic)

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
    def test_drone_keyboard_gazebo_with_KF(self):
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
            'mathias_controller_with_KF': True,
            'mathias_controller_config_file_path_with_extension':
                f'{os.environ["CODEDIR"]}/src/sim/ros/config/actor/mathias_controller_with_KF.yml',
            'yaw_or': 0,
            'x_pos': 0,
            'y_pos': 0
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
            TopicConfig(topic_name=self.visualisation_topic,
                        msg_type='Image')
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
                            True, 30, 0.1, ros_topic=self.ros_topic)
        measured_data = {}
        index = 0
        while True:
            # publish reset
            self.ros_topic.publishers['/fsm/reset'].publish(Empty())

            self._unpause_client.wait_for_service()
            self._unpause_client.call()

            # index = self.tweak_combined_axis_keyboard(measured_data, index, [0, 0, 0])
            index = self.tweak_combined_axis_keyboard(measured_data, index, [3, 1, 0])

    @unittest.skip
    def test_drone_relative_positioning_real_bebop_with_KF(self):
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
            'mathias_controller_with_KF': True,
            'mathias_controller_config_file_path_with_extension':
                f'{os.environ["CODEDIR"]}/src/sim/ros/config/actor/mathias_controller_with_KF_real_bebop.yml'
        }

        # spinoff roslaunch
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=self._config,
                                       visible=True)
        self.visualisation_topic = '/actor/mathias_controller/visualisation'
        subscribe_topics = [
            TopicConfig(topic_name=rospy.get_param('/robot/position_sensor/topic'),
                        msg_type=rospy.get_param('/robot/position_sensor/type')),
            TopicConfig(topic_name='/fsm/state',
                        msg_type='String'),
            TopicConfig(topic_name=self.visualisation_topic,
                        msg_type='Image')
        ]

        self._reference_topic = '/reference_pose'
        self._reference_type = 'PointStamped'
        publish_topics = [
            TopicConfig(topic_name='/fsm/reset', msg_type='Empty'),
            TopicConfig(topic_name=self._reference_topic, msg_type=self._reference_type),
            TopicConfig(topic_name='/bebop/land', msg_type='Empty'),
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
            # index = self.tweak_steady_pose(measured_data, index)
            # index = self.tweak_separate_axis_keyboard(measured_data, index, axis=1)
#            index = self.tweak_combined_axis_keyboard(measured_data, index, point=[0., 0., 0.])
            #index = self.tweak_combined_axis_keyboard(measured_data, index, point=[0., 0., 0.5])
            index = self.tweak_combined_axis_keyboard(measured_data, index, point=[3, 1, 1])
            index = self.tweak_combined_axis_keyboard(measured_data, index, point=[3, 1, -1])

 #   @unittest.skip
    def test_waypoints_tracking_in_gazebo_with_KF_with_keyboard(self):
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

        self._config = {
            'output_path': self.output_dir,
            'world_name': 'gate_test', # 'hexagon',
            'robot_name': 'drone_sim',
            'gazebo': True,
            'fsm': True,
            'fsm_mode': 'TakeOverRun',
            'control_mapping': True,
            'control_mapping_config': 'mathias_controller_keyboard',
            'waypoint_indicator': True,
            'altitude_control': False,
            'mathias_controller_with_KF': True,
            'starting_height': 1.,
            'keyboard': True,
            'mathias_controller_config_file_path_with_extension':
                f'{os.environ["CODEDIR"]}/src/sim/ros/config/actor/mathias_controller_with_KF.yml',
            'yaw_or': 0,
            'x_pos': 0,
            'y_pos': 0
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
        self._unpause_client = rospy.ServiceProxy('/gazebo/unpause_physics', Emptyservice)
        self._pause_client = rospy.ServiceProxy('/gazebo/pause_physics', Emptyservice)
        self._set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self._unpause_client.wait_for_service()
        self._unpause_client.call()
        safe_wait_till_true('"/fsm/state" in kwargs["ros_topic"].topic_values.keys()',
                            True, 15, 0.1, ros_topic=self.ros_topic)
        self.assertEqual(self.ros_topic.topic_values['/fsm/state'].data, FsmState.Unknown.name)
        self._pause_client.wait_for_service()
        self._pause_client.call()

        while True:
            # publish reset
            self.ros_topic.publishers['/fsm/reset'].publish(Empty())

            self._unpause_client.wait_for_service()
            self._unpause_client.call()

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
                print(self.ros_topic.topic_values["/fsm/state"].data)

            # see it reaches the goal state:
#            safe_wait_till_true('kwargs["ros_topic"].topic_values["/fsm/state"].data',
#                                FsmState.Terminated.name, 20, 0.1, ros_topic=self.ros_topic)

            self._pause_client.wait_for_service()
            self._pause_client.call()

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

    @unittest.skip
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

    @unittest.skip
    def test_april_tag_detector_real_bebop(self):
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

        self._config = {
            'output_path': self.output_dir,
            'world_name': 'april_tag',
            'robot_name': 'bebop_real',
            'gazebo': False,
            'april_tag_detector': True,
        }

        # spinoff roslaunch
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=self._config,
                                       visible=True)

        # subscribe to command control
        subscribe_topics = [
            TopicConfig(topic_name='/reference_ground_point', msg_type='PointStamped'),
        ]

        self.ros_topic = TestPublisherSubscriber(
            subscribe_topics=subscribe_topics,
            publish_topics=[]
        )
        safe_wait_till_true('"/reference_ground_point" in kwargs["ros_topic"].topic_values.keys()',
                            True, 25, 0.1, ros_topic=self.ros_topic)
        while True:
            print(f'waypoint: {self.ros_topic.topic_values["/reference_ground_point"]}')
            time.sleep(0.5)

    @unittest.skip
    def test_april_tag_detector_real_bebop_KF(self):
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

        self._config = {
            'output_path': self.output_dir,
            'world_name': 'april_tag',
            'robot_name': 'bebop_real',
            'gazebo': False,
            'fsm': True,
            'fsm_mode': 'TakeOverRun',
            'control_mapping': True,
            'control_mapping_config': 'mathias_controller_keyboard',
            'april_tag_detector': True,
            'altitude_control': False,
            'mathias_controller_with_KF': True,
            'starting_height': 1.,
            'keyboard': True,
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
            TopicConfig(topic_name='/reference_pose', msg_type='PointStamped'),
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

        index = 0
        while True:
            print(f'start loop: {index} with resetting')
            # publish reset
            self.ros_topic.publishers['/fsm/reset'].publish(Empty())
            rospy.sleep(0.5)

            print(f'waiting in overtake state')
            while self.ros_topic.topic_values["/fsm/state"].data != FsmState.Running.name:
                rospy.sleep(0.5)

            #safe_wait_till_true('"/reference_ground_point" in kwargs["ros_topic"].topic_values.keys()',
            #                    True, 10, 0.1, ros_topic=self.ros_topic)
            waypoints = []
            print(f'waiting in running state')
            while self.ros_topic.topic_values["/fsm/state"].data != FsmState.TakenOver.name:
                if '/reference_pose' in self.ros_topic.topic_values.keys() \
                        and '/bebop/odom' in self.ros_topic.topic_values.keys():
                    odom = self.ros_topic.topic_values[rospy.get_param('/robot/position_sensor/topic')]
                    point = transform([self.ros_topic.topic_values['/reference_pose'].point],
                                      orientation=odom.pose.pose.orientation,
                                      translation=odom.pose.pose.position)[0]
                    waypoints.append(point)
                rospy.sleep(0.5)
            if len(waypoints) != 0:
                plt.clf()
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter([_.x for _ in waypoints],
                           [_.y for _ in waypoints],
                           [_.z for _ in waypoints], label='waypoints')
                ax.legend()
                plt.savefig(os.path.join(self.output_dir, f'image_{index}.jpg'))
                #plt.show()
            index += 1

    @unittest.skip
    def test_april_tag_detector_gazebo_KF(self):
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

        self._config = {
            'output_path': self.output_dir,
            'world_name': 'april_tag',
            'robot_name': 'drone_sim_down_cam',
            'gazebo': True,
            'fsm': True,
            'fsm_mode': 'TakeOverRun',
            'control_mapping': True,
            'control_mapping_config': 'mathias_controller_keyboard',
            'april_tag_detector': True,
            'altitude_control': False,
            'mathias_controller_with_KF': True,
            'starting_height': 1.,
            'keyboard': True,
            'mathias_controller_config_file_path_with_extension':
                f'{os.environ["CODEDIR"]}/src/sim/ros/config/actor/mathias_controller_with_KF.yml',
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
            TopicConfig(topic_name='/reference_pose', msg_type='PointStamped'),
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
                            True, 235, 0.1, ros_topic=self.ros_topic)
        self.assertEqual(self.ros_topic.topic_values['/fsm/state'].data, FsmState.Unknown.name)

        index = 0
        while True:
            print(f'start loop: {index} with resetting')
            # publish reset
            self.ros_topic.publishers['/fsm/reset'].publish(Empty())
            rospy.sleep(0.5)

            print(f'waiting in overtake state')
            while self.ros_topic.topic_values["/fsm/state"].data != FsmState.Running.name:
                rospy.sleep(0.5)

            # safe_wait_till_true('"/reference_ground_point" in kwargs["ros_topic"].topic_values.keys()',
            #                    True, 10, 0.1, ros_topic=self.ros_topic)
            waypoints = []
            print(f'waiting in running state')
            while self.ros_topic.topic_values["/fsm/state"].data != FsmState.TakenOver.name:
                if '/reference_pose' in self.ros_topic.topic_values.keys() \
                        and '/bebop/odom' in self.ros_topic.topic_values.keys():
                    odom = self.ros_topic.topic_values[rospy.get_param('/robot/position_sensor/topic')]
                    point = transform([self.ros_topic.topic_values['/reference_pose'].point],
                                      orientation=odom.pose.pose.orientation,
                                      translation=odom.pose.pose.position)[0]
                    waypoints.append(point)
                rospy.sleep(0.5)
            if len(waypoints) != 0:
                plt.clf()
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter([_.x for _ in waypoints],
                           [_.y for _ in waypoints],
                           [_.z for _ in waypoints], label='waypoints')
                ax.legend()
                plt.savefig(os.path.join(self.output_dir, f'image_{index}.jpg'))
                # plt.show()
            index += 1

    def tearDown(self) -> None:
        print(f'shutting down...')
        self._ros_process.terminate()
        # shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
