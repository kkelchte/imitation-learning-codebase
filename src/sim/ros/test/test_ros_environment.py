import os
import shutil
import subprocess
import time
import unittest
import warnings

from datetime import datetime
import shlex

import rospy
import yaml
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Empty

from src.sim.common.actors import ActorConfig
from src.sim.common.data_types import EnvironmentType, ActorType, TerminalType
from src.sim.common.environment import EnvironmentConfig, RosConfig, RosLaunchConfig
from src.sim.ros.catkin_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.src.process_wrappers import RosWrapper, ProcessState
from src.sim.ros.src.ros_environment import RosEnvironment
from src.sim.ros.test.common_utils import TestPublisherSubscriber, TopicConfig, \
    get_fake_image, get_fake_odometry, get_fake_laser_scan
from src.core.utils import camelcase_to_snake_format

"""
test starting ros_environment in subprocess and interact with it through publishing & subscribing topics
"""

# warnings.filterwarnings("ignore")


class TestRosEnvironment(unittest.TestCase):

    def start_test(self, gazebo: bool = False) -> None:
        self.output_dir = f'tmp_test_dir/{datetime.strftime(datetime.now(), format="%y-%m-%d_%H-%M-%S")}'
        os.makedirs(self.output_dir, exist_ok=True)
        # config = EnvironmentConfig(
        #     output_path=self.output_dir,
        #     factory_key=EnvironmentType.Ros,
        #     max_number_of_steps=300,
        #     ros_config=RosConfig(
        #         visible_xterm=True,
        #         ros_launch_config=RosLaunchConfig(
        #             random_seed=123,
        #             gazebo=gazebo,
        #             world_name='object_world',
        #             robot_name='turtlebot_sim',
        #             fsm=True,
        #             fsm_config='single_run',
        #             control_mapping=True,
        #             control_mapping_config='test_ros_environment',
        #         )
        #     ),
        #     actor_configs=[
        #         ActorConfig(
        #             name='keyboard',
        #             type=ActorType.User,
        #             file=f'src/sim/ros/config/keyboard/turtlebot_sim.yml'
        #         ),
        #         ActorConfig(
        #             name='ros_expert',
        #             type=ActorType.Expert,
        #             file=f'src/sim/ros/config/actor/ros_expert.yml'
        #         ),
        #     ]
        # )

        with open('src/sim/ros/test/config/test_ros_environment.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config['output_path'] = self.output_dir
        with open(os.path.join(self.output_dir, 'config.yml'), 'w') as f:
            yaml.dump(config, f)

        self._ros_environment_process = subprocess.Popen(
            shlex.split(
                f'python3.7 src/sim/ros/test/launch_ros.py --config {self.output_dir}/config.yml'
            )
        )

    @unittest.skip
    def test_ros_launch_in_popen(self):
        self.start_test()
        self.assertTrue(self._ros_environment_process.poll() is None)
        time.sleep(10)

    def load_ros_topic_with_fake_sensors_and_control_commands(self):
        # fake command publish topics
        self.command_topic = rospy.get_param('/robot/command_topic')
        self.supervision_topic = rospy.get_param('/control_mapping/supervision_topic')
        subscribe_topics = [
            TopicConfig(topic_name=self.command_topic, msg_type="Twist"),
            TopicConfig(topic_name=self.supervision_topic, msg_type="Twist"),
            TopicConfig(topic_name=rospy.get_param('/fsm/state_topic'), msg_type="String"),
            TopicConfig(topic_name='/ros_environment/state', msg_type="RosState"),
        ]
        for sensor in rospy.get_param('/robot/sensors', []):
            sensor_topic = rospy.get_param(f'/robot/{sensor}_topic')
            sensor_type = rospy.get_param(f'/robot/{sensor}_type')
            subscribe_topics.append(
                TopicConfig(topic_name=sensor_topic, msg_type=sensor_type)
            )

        # create publishers for all control topics < control_mapper/default.yml
        self._mapping = rospy.get_param('/control_mapping/mapping')
        publish_topics = [
            TopicConfig(topic_name=rospy.get_param('/fsm/state_topic'), msg_type='String')
        ]
        self._control_topics = []
        for state, mode in self._mapping.items():
            if 'command' in mode.keys():
                publish_topics.append(
                    TopicConfig(topic_name=mode['command'], msg_type='Twist')
                )
                self._control_topics.append(mode['command'])
            if 'supervision' in mode.keys():
                publish_topics.append(
                    TopicConfig(topic_name=mode['supervision'], msg_type='Twist')
                )
                self._control_topics.append(mode['supervision'])
        self._sensor_topics = []
        for sensor in rospy.get_param('/robot/sensors', []):
            sensor_topic = rospy.get_param(f'/robot/{sensor}_topic')
            sensor_type = rospy.get_param(f'/robot/{sensor}_type')
            publish_topics.append(
                TopicConfig(topic_name=sensor_topic, msg_type=sensor_type)
            )
            self._sensor_topics.append((sensor_topic, sensor_type))

        self.ros_topic = TestPublisherSubscriber(
            subscribe_topics=subscribe_topics,
            publish_topics=publish_topics
        )

    def publish_on_all_sensors(self, number_of_times: int = 1):
        for _ in range(number_of_times):
            for sensor_topic, sensor_type in self._sensor_topics:
                self.ros_topic.publishers[sensor_topic].publish(
                    eval(f'get_fake_{camelcase_to_snake_format(sensor_type)}()')
                )
            time.sleep(0.1)

    # @unittest.skip
    def test_ros_environment_without_gazebo(self):
        self.start_test(gazebo=False)
        time.sleep(6)
        self.load_ros_topic_with_fake_sensors_and_control_commands()

        # publish on sensor + actor topics => assert they are retrieved in state
        self.ros_topic.publishers[rospy.get_param('/fsm/state_topic')].publish(f'{FsmState.Running.name}')
        time.sleep(0.1)
        self.publish_on_all_sensors(number_of_times=20)
        time.sleep(1)
        self.assertTrue('/ros_environment/state' in self.ros_topic.topic_values.keys())
        print(self._sensor_topics)

        # publish fake collision => assert it is stopped with failure

        # publish fake waypoints reached => assert waypoint changed => assert goal state is reached and stopped

        # take a few steps

    def tearDown(self) -> None:
        self._ros_environment_process.terminate()
        while self._ros_environment_process.poll() is None:
            time.sleep(0.1)
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
