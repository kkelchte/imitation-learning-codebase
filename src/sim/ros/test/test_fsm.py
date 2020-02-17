import unittest
import time
from typing import List

import rospy
from dataclasses import dataclass

from nav_msgs.msg import Odometry
from std_msgs.msg import String, Empty  # Do not remove!
from sensor_msgs.msg import LaserScan  # Do not remove!

from src.sim.ros.catkin_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.common.data_types import TerminalType, ProcessState
from src.sim.ros.src.process_wrappers import RosWrapper

""" Test FSM in different modes

Note that due to the restriction of ROS to run only one rospy.init_node per process,
only one test can be run at a time.
"""


@dataclass
class TopicConfig:
    name: str
    msg_type: str


class TestPublisherSubscriber:

    def __init__(self, topics: List[TopicConfig]):
        self.topic_values = {}
        for topic_config in topics:
            rospy.Subscriber(topic_config.name,
                             eval(topic_config.msg_type),
                             self._store,
                             callback_args=topic_config.name)
        self._set_publishers()
        rospy.init_node(f'test_fsm', anonymous=True)

    def _set_publishers(self):
        if rospy.has_param('/robot/depth_scan_topic'):
            self.scan_publisher = rospy.Publisher(rospy.get_param('/robot/depth_scan_topic'),
                                                  eval(rospy.get_param('/robot/depth_scan_type')),
                                                  queue_size=10)
        if rospy.has_param('/robot/odometry_topic'):
            self.odom_publisher = rospy.Publisher(rospy.get_param('/robot/odometry_topic'),
                                                  eval(rospy.get_param('/robot/odometry_type')),
                                                  queue_size=10)
        if rospy.has_param('/fsm/go_topic'):
            self.go_publisher = rospy.Publisher(rospy.get_param('/fsm/go_topic'), Empty, queue_size=10)
        if rospy.has_param('/fsm/overtake_topic'):
            self.overtake_publisher = rospy.Publisher(rospy.get_param('/fsm/overtake_topic'), Empty, queue_size=10)
        if rospy.has_param('/fsm/finish_topic'):
            self.finish_publisher = rospy.Publisher(rospy.get_param('/fsm/finish_topic'), Empty, queue_size=10)
        if rospy.has_param('/fsm/reset_topic'):
            self.reset_publisher = rospy.Publisher(rospy.get_param('/fsm/reset_topic'), Empty, queue_size=10)

    def _store(self, msg, field_name: str):
        self.topic_values[field_name] = msg.data

    def publish_fake_collision_on_scan(self):
        scan = LaserScan()
        scan.ranges = [0.2]*360
        self.scan_publisher.publish(scan)

    def publish_fake_odom(self, x: float = 0, y: float = 0, z: float = 0):
        odom = Odometry()
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = z
        self.odom_publisher.publish(odom)

    def publish_fake_go(self):
        self.go_publisher.publish(Empty())

    def publish_fake_takeover(self):
        self.overtake_publisher.publish(Empty())

    def publish_fake_finish(self):
        self.finish_publisher.publish(Empty())

    def publish_fake_reset(self):
        self.reset_publisher.publish(Empty())


class TestFsm(unittest.TestCase):

    def start_test(self, config: dict) -> None:
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=config,
                                       visible=True)
        self.state_topic = rospy.get_param('/fsm/state_topic')
        self.terminal_topic = rospy.get_param('/fsm/terminal_topic')
        topics = [
            TopicConfig(name=rospy.get_param('/fsm/terminal_topic'), msg_type="String"),
            TopicConfig(name=rospy.get_param('/fsm/state_topic'), msg_type="String"),
        ]
        self.ros_topic = TestPublisherSubscriber(topics)

    @unittest.skip
    def test_single_run(self):
        config = {
            'robot_name': 'drone_sim',
            'fsm': True,
            'fsm_config': 'single_run'
        }
        self.start_test(config=config)
        time.sleep(1)
        self.ros_topic.publish_fake_reset()
        time.sleep(rospy.get_param('/world/delay_evaluation') + 0.5)
        time.sleep(1)
        self.ros_topic.publish_fake_collision_on_scan()
        time.sleep(1)
        self.assertEqual(self.ros_topic.topic_values[self.state_topic], FsmState.Terminated.name)
        self.assertEqual(self.ros_topic.topic_values[self.terminal_topic], TerminalType.Failure.name)
        self.stop_test()

    @unittest.skip
    def test_takeoff_run(self):
        config = {
            'robot_name': 'drone_sim',
            'fsm': True,
            'fsm_config': 'takeoff_run'
        }
        self.start_test(config=config)
        time.sleep(1)
        self.ros_topic.publish_fake_reset()
        time.sleep(rospy.get_param('/world/delay_evaluation') + 0.5)
        self.ros_topic.publish_fake_odom(x=0, y=0, z=4)
        time.sleep(rospy.get_param('/world/delay_evaluation')+1)
        self.assertEqual(self.ros_topic.topic_values[self.state_topic], FsmState.Running.name)
        self.ros_topic.publish_fake_odom(x=1, y=1, z=4)
        time.sleep(1)
        self.ros_topic.publish_fake_odom(x=100, y=100, z=4)
        time.sleep(1)
        self.assertEqual(self.ros_topic.topic_values[self.terminal_topic], TerminalType.Success.name)
        self.stop_test()

    @unittest.skip
    def test_takeover_run(self):
        config = {
            'robot_name': 'drone_sim',
            'fsm': True,
            'fsm_config': 'takeover_run'
        }
        self.start_test(config=config)
        time.sleep(1)
        self.ros_topic.publish_fake_reset()
        time.sleep(rospy.get_param('/world/delay_evaluation') + 0.5)
        self.ros_topic.publish_fake_go()
        time.sleep(rospy.get_param('/world/delay_evaluation')+1)
        self.assertEqual(self.ros_topic.topic_values[self.state_topic], FsmState.Running.name)
        self.ros_topic.publish_fake_takeover()
        time.sleep(1)
        self.assertEqual(self.ros_topic.topic_values[self.state_topic], FsmState.TakenOver.name)
        self.ros_topic.publish_fake_finish()
        time.sleep(1)
        self.assertEqual(self.ros_topic.topic_values[self.terminal_topic], TerminalType.Unknown.name)
        self.stop_test()

    @unittest.skip
    def test_takeover_run_driveback(self):
        config = {
            'robot_name': 'drone_sim',
            'fsm': True,
            'fsm_config': 'takeover_run_driveback'
        }
        self.start_test(config=config)
        time.sleep(1)
        self.ros_topic.publish_fake_reset()
        time.sleep(rospy.get_param('/world/delay_evaluation') + 0.5)
        self.ros_topic.publish_fake_go()
        time.sleep(rospy.get_param('/world/delay_evaluation')+1)
        self.assertEqual(self.ros_topic.topic_values[self.state_topic], FsmState.Running.name)
        self.ros_topic.publish_fake_collision_on_scan()
        time.sleep(1)
        self.assertEqual(self.ros_topic.topic_values[self.state_topic], FsmState.DriveBack.name)
        self.assertEqual(self.ros_topic.topic_values[self.terminal_topic], TerminalType.Failure.name)
        self.ros_topic.publish_fake_go()
        time.sleep(1)
        self.assertEqual(self.ros_topic.topic_values[self.state_topic], FsmState.Running.name)
        self.stop_test()

    @unittest.skip
    def test_multiple_runs(self):
        config = {
            'robot_name': 'drone_sim',
            'fsm': True,
            'fsm_config': 'single_run'
        }
        self.start_test(config=config)
        time.sleep(rospy.get_param('/world/delay_evaluation'))
        self.ros_topic.publish_fake_reset()
        time.sleep(rospy.get_param('/world/delay_evaluation') + 0.5)
        self.ros_topic.publish_fake_collision_on_scan()
        time.sleep(1)
        self.assertEqual(self.ros_topic.topic_values[self.state_topic], FsmState.Terminated.name)
        self.assertEqual(self.ros_topic.topic_values[self.terminal_topic], TerminalType.Failure.name)
        self.ros_topic.publish_fake_reset()
        time.sleep(rospy.get_param('/world/delay_evaluation') + 0.5)
        self.assertEqual(self.ros_topic.topic_values[self.state_topic], FsmState.Running.name)
        self.ros_topic.publish_fake_collision_on_scan()
        time.sleep(1)
        self.assertEqual(self.ros_topic.topic_values[self.state_topic], FsmState.Terminated.name)
        self.assertEqual(self.ros_topic.topic_values[self.terminal_topic], TerminalType.Failure.name)
        self.stop_test()

    @unittest.skip
    def test_success_by_reaching_goal(self):
        config = {
            'robot_name': 'drone_sim',
            'fsm': True,
            'fsm_config': 'single_run',
            'world_name': 'object_world'
        }
        self.start_test(config=config)
        time.sleep(rospy.get_param('/world/delay_evaluation'))
        self.ros_topic.publish_fake_reset()
        time.sleep(rospy.get_param('/world/delay_evaluation') + 0.5)

        self.ros_topic.publish_fake_odom(x=2, y=2, z=0.5)
        time.sleep(1)
        self.assertEqual(self.ros_topic.topic_values[self.terminal_topic], TerminalType.Success.name)
        self.stop_test()

    def stop_test(self) -> None:
        self.assertEqual(ProcessState.Terminated,
                         self._ros_process.terminate())


if __name__ == '__main__':
    unittest.main()
