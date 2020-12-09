import os
import shutil
import time
import unittest

import rospy
from geometry_msgs.msg import Twist

from src.core.utils import get_filename_without_extension, get_data_dir
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.test.common_utils import TestPublisherSubscriber, TopicConfig

"""
For each FSM state, test correct mapping of control.
"""


class TestControlMapper(unittest.TestCase):

    def start(self) -> None:
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        config = {
            'robot_name': 'test_control_mapper',
            'world_name': 'test_control_mapper',
            'fsm': False,
            'control_mapping': True,
            'control_mapping_config': 'test_control_mapper',
            'output_path': self.output_dir,
            'waypoint_indicator': False
        }

        # spinoff roslaunch
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=config,
                                       visible=False)
        # subscribe to robot control topics
        self.command_topics = {
            'command_a': rospy.get_param('/robot/command_a_topic'),
            'command_b': rospy.get_param('/robot/command_b_topic'),
            'command_c': rospy.get_param('/robot/command_c_topic')
        }
        subscribe_topics = [
            TopicConfig(topic_name=topic_name, msg_type="Twist") for topic_name in self.command_topics.values()
        ]
        # create publishers for all control topics < control_mapper/default.yml
        self._mapping = rospy.get_param('/control_mapping/mapping')

        publish_topics = [
            TopicConfig(topic_name='/fsm/state', msg_type='String')
        ]
        self._control_topics = [mode[key] for mode in self._mapping.values()
                                for key in mode.keys()]
        self._control_topics = set(self._control_topics)

        self.ros_topic = TestPublisherSubscriber(
            subscribe_topics=subscribe_topics,
            publish_topics=publish_topics + [TopicConfig(topic_name=name, msg_type='Twist')
                                             for name in self._control_topics]
        )

    # @unittest.skip
    def test_control_mapper(self):
        self.start()
        # for each fsm state
        for fsm_state in [FsmState.Unknown,
                          FsmState.Running,
                          FsmState.DriveBack,
                          FsmState.TakenOver,
                          FsmState.Terminated]:
            print(f'FSM STATE: {fsm_state}')
            #   publish fsm state
            self.ros_topic.publishers['/fsm/state'].publish(fsm_state.name)
            #   wait
            time.sleep(0.5)
            #   publish on all controls
            solution = {}
            for index, control_topic in enumerate(list(set(self._control_topics))):
                control_command = Twist()
                control_command.linear.x = index * 100
                solution[control_topic] = control_command.linear.x
                self.ros_topic.publishers[control_topic].publish(control_command)
            #   wait
            time.sleep(0.5)
            print(f'published actor topics: {solution}')
            #   assert control is equal to intended control
            for control_type, actor_topic in self._mapping[fsm_state.name].items():
                received_control = self.ros_topic.topic_values[self.command_topics[control_type]]
                self.assertEqual(received_control.linear.x, solution[actor_topic])

    def tearDown(self) -> None:
        self._ros_process.terminate()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
