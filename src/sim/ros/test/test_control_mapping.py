import time
import unittest

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty

from src.sim.ros.catkin_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.src.process_wrappers import RosWrapper, ProcessState
from src.sim.ros.test.common_utils import TestPublisherSubscriber, TopicConfig

"""
For each FSM state, test correct mapping of control.
"""


class TestControlMapper(unittest.TestCase):

    def start_test(self) -> None:
        config = {
            'robot_name': 'drone_sim',
            'fsm': False,
            'control_mapper': True,
            'control_mapper_config': 'test'
        }
        # spinoff roslaunch
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=config,
                                       visible=True)
        # subscribe to supervision and command control
        self.command_topic = rospy.get_param('/robot/command_topic')
        self.supervision_topic = rospy.get_param('/control_mapping/supervision_topic')
        subscribe_topics = [
            TopicConfig(topic_name=self.command_topic, msg_type="Twist"),
            TopicConfig(topic_name=self.supervision_topic, msg_type="Twist"),
            TopicConfig(topic_name='/fsm_state', msg_type="String"),
        ]
        # create publishers for all control topics < control_mapper/default.yml
        self._mapping = rospy.get_param('/control_mapping/mapping')
        publish_topics = [
            TopicConfig(topic_name='/fsm_state', msg_type='String')
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
        self.ros_topic = TestPublisherSubscriber(
            subscribe_topics=subscribe_topics,
            publish_topics=publish_topics
        )

    def test_control_mapper(self):
        self.start_test()
        # for each fsm state
        for fsm_state in [FsmState.Running,
                          FsmState.DriveBack,
                          FsmState.TakenOver,
                          FsmState.TakeOff]:
            print(f'FSM STATE: {fsm_state}')
            # fsm_state = FsmState.Running
            #   publish fsm state
            self.ros_topic.publishers['/fsm_state'].publish(fsm_state.name)
            #   wait
            time.sleep(1)
            #   publish on all controls
            solution = {}
            for index, control_topic in enumerate(list(set(self._control_topics))):
                control_command = Twist()
                control_command.linear.x = index * 100
                solution[control_topic] = control_command.linear.x
                self.ros_topic.publishers[control_topic].publish(control_command)
            #   wait
            time.sleep(1)
            #   assert control is equal to intended control
            if 'command' in self._mapping[fsm_state.name].keys():
                original_topic = self._mapping[fsm_state.name]['command']
                received_control = self.ros_topic.topic_values[self.command_topic]
                self.assertEqual(received_control.linear.x, solution[original_topic])
            if 'supervision' in self._mapping[fsm_state.name].keys():
                original_topic = self._mapping[fsm_state.name]['supervision']
                received_control = self.ros_topic.topic_values[self.supervision_topic]
                self.assertEqual(received_control.linear.x, solution[original_topic])

    def tearDown(self) -> None:
        self.assertEqual(self._ros_process.terminate(),
                         ProcessState.Terminated)


if __name__ == '__main__':
    unittest.main()
