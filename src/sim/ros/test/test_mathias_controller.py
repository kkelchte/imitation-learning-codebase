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
from std_msgs.msg import Empty
from std_srvs.srv import Empty as Emptyservice, EmptyRequest
import matplotlib.pyplot as plt

from src.core.utils import get_filename_without_extension, get_to_root_dir, get_data_dir, safe_wait_till_true
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.test.common_utils import TopicConfig, TestPublisherSubscriber, get_fake_laser_scan


class TestMathiasController(unittest.TestCase):

    def start(self):
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

    @unittest.skip
    def test_node_without_gazebo(self):
        self.start()
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

    #@unittest.skip
    def test_drone_positioning(self) -> None:
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
                                FsmState.Running.name, 45, 0.1, ros_topic=self.ros_topic)
            # tweak z, x, y parameters
            #self.tweak_separate_axis(index)

            # tweak joint trajectory

    def tweak_separate_axis(self, index):
        measured_data = {}
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

    def tearDown(self) -> None:
        self._ros_process.terminate()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
