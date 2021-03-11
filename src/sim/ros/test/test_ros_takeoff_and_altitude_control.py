import os
import shutil
import unittest

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose
from std_msgs.msg import Empty
from std_srvs.srv import Empty as Emptyservice

from src.core.utils import get_filename_without_extension, get_to_root_dir, get_data_dir, safe_wait_till_true
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.src.utils import quaternion_from_euler
from src.sim.ros.test.common_utils import TopicConfig, TestPublisherSubscriber


class TestTakeOffAndAltitudeControl(unittest.TestCase):

    @unittest.skip
    def test_single_drone(self) -> None:
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

        height = 5
        self._config = {
            'output_path': self.output_dir,
            'world_name': 'empty',
            'robot_name': 'drone_sim',
            'gazebo': True,
            'fsm': True,
            'fsm_mode': 'TakeOverRun',
            'control_mapping': True,
            'control_mapping_config': 'ros_expert',
            'altitude_control': True,
            'ros_expert': True,
            'starting_height': height
        }

        # spinoff roslaunch
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=self._config,
                                       visible=False)

        # subscribe to command control
        subscribe_topics = [
            TopicConfig(topic_name=rospy.get_param('/robot/position_sensor/topic'),
                        msg_type=rospy.get_param('/robot/position_sensor/type')),
            TopicConfig(topic_name='/fsm/state',
                        msg_type='String')
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

        # pause again before start
        self._pause_client.wait_for_service()
        self._pause_client.call()

        self._set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        for _ in range(3):
            print(f'round: {_}')

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
            # check current height
            z_pos = self.ros_topic.topic_values[rospy.get_param('/robot/position_sensor/topic')].pose.pose.position.z
            print(f'final_height: {z_pos}')
            self.assertLess(abs(z_pos - height), 0.2)

            self._pause_client.wait_for_service()
            self._pause_client.call()

    #@unittest.skip
    def test_double_drone(self) -> None:
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

        height = 5
        self._config = {
            'output_path': self.output_dir,
            'world_name': 'empty',
            'robot_name': 'double_drone_sim',
            'gazebo': True,
            'fsm': True,
            'fsm_mode': 'TakeOverRun',
            'control_mapping': True,
            'control_mapping_config': 'python_adversarial',
            'altitude_control': True,
            'ros_expert': True,
            'starting_height': height
        }

        # spinoff roslaunch
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=self._config,
                                       visible=False)

        # subscribe to command control
        subscribe_topics = [
            TopicConfig(topic_name=rospy.get_param('/robot/tracking_position_sensor/topic'),
                        msg_type=rospy.get_param('/robot/tracking_position_sensor/type')),
            TopicConfig(topic_name=rospy.get_param('/robot/fleeing_position_sensor/topic'),
                        msg_type=rospy.get_param('/robot/fleeing_position_sensor/type')),
            TopicConfig(topic_name='/fsm/state',
                        msg_type='String')
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
        self.assertEqual(self.ros_topic.topic_values['/fsm/state'].data,
                         FsmState.Unknown.name)

        # pause again before start
        self._pause_client.wait_for_service()
        self._pause_client.call()

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

        for _ in range(3):
            print(f'round: {_}')
            for model_name in rospy.get_param('/robot/model_name'):
                set_model_state(model_name)

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

            # check current height
            for agent in ['tracking', 'fleeing']:
                z_pos = self.ros_topic.topic_values[rospy.get_param(f'/robot/{agent}_position_sensor/topic')].pose.position.z
                self.assertLess(abs(z_pos - height), 0.2)

    def tearDown(self) -> None:
        self._ros_process.terminate()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
