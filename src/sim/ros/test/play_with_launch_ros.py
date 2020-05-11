import os
import shutil
import subprocess
import time
import unittest

import shlex

import rospy
import yaml

from src.sim.ros.test.common_utils import TestPublisherSubscriber, TopicConfig
from src.core.utils import camelcase_to_snake_format, get_filename_without_extension

"""
test starting ros_environment in subprocess and interact with it through publishing & subscribing topics
validate by checking ros_environment/state topic
"""


class TestRosEnvironment(unittest.TestCase):

    def start_test(self, config_file: str) -> None:
        self.output_dir = f'test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        with open(f'src/sim/ros/test/config/{config_file}.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config['output_path'] = self.output_dir
        with open(os.path.join(self.output_dir, 'config.yml'), 'w') as f:
            yaml.dump(config, f)

        self._ros_environment_process = subprocess.Popen(
            shlex.split(
                f'python3.7 src/sim/ros/test/launch_ros.py --config {self.output_dir}/config.yml'
            )
        )

    def load_ros_topic_with_sensors_and_control_commands(self):
        time.sleep(5)
        while not rospy.has_param('/robot/command_topic'):
            time.sleep(0.1)

        # fake command publish topics
        self.command_topic = rospy.get_param('/robot/command_topic')
        # self.supervision_topic = rospy.get_param('/control_mapping/supervision_topic')
        subscribe_topics = [
            TopicConfig(topic_name=self.command_topic, msg_type="Twist"),
            # TopicConfig(topic_name=self.supervision_topic, msg_type="Twist"),
            # TopicConfig(topic_name=rospy.get_param('/fsm/state_topic'), msg_type="String"),
            TopicConfig(topic_name='/ros_environment/state', msg_type="RosState"),
        ]
        for sensor in rospy.get_param('/robot/sensors', []):
            sensor_topic = rospy.get_param(f'/robot/{sensor}_topic')
            sensor_type = rospy.get_param(f'/robot/{sensor}_type')
            subscribe_topics.append(
                TopicConfig(topic_name=sensor_topic, msg_type=sensor_type)
            )

        # create publishers for all control topics < control_mapper/default.yml
        self._mapping = rospy.get_param('/control_mapping/mapping', {})
        publish_topics = [
            # TopicConfig(topic_name=rospy.get_param('/fsm/state_topic'), msg_type='String')
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

    def test_ros_environment_barebones_gazebo(self):
        self.start_test(config_file='test_empty_config')
        self.load_ros_topic_with_sensors_and_control_commands()

        # wait for the first publish ros_environment state message to know it has started correctly
        while '/ros_environment/state' not in self.ros_topic.topic_values.keys():
            time.sleep(0.01)

        # for each sensor publish fake message and validate ros_environment/state, sleep 0.2s to ensure update
        for sensor_topic, sensor_type in self._sensor_topics:
            msg = eval(f'get_fake_{camelcase_to_snake_format(sensor_type)}()')
            self.ros_topic.publishers[sensor_topic].publish(msg)
            # time.sleep(5)  # should be enough to take one step but not two at a rate of 10fps
            while len(self.ros_topic.last_received_sensor_readings) == 0:
                time.sleep(0.1)
            sensor_data = self.ros_topic.last_received_sensor_readings
            self.assertEqual(len(sensor_data), 1)
            sensor_msg = eval(f'sensor_data[0].{camelcase_to_snake_format(sensor_type)}')
            self.assertTrue(isinstance(sensor_msg, eval(sensor_type)))
            self.assertTrue(eval(f'compare_{camelcase_to_snake_format(sensor_type)}(msg, sensor_msg)'))

    def tearDown(self) -> None:
        self._ros_environment_process.terminate()
        while self._ros_environment_process.poll() is None:
            time.sleep(0.1)
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
