import os
import shutil
import unittest
import time
from typing import List

import rospy
from dataclasses import dataclass
import numpy as np

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty, Float32, String  # Do not remove!
from sensor_msgs.msg import LaserScan  # Do not remove!

from imitation_learning_ros_package.msg import RosReward

from src.core.utils import get_filename_without_extension, get_data_dir, safe_wait_till_true
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.core.data_types import TerminationType, SensorType
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.test.common_utils import TopicConfig, TestPublisherSubscriber, get_fake_odometry, get_fake_laser_scan, \
    get_fake_modified_state

""" Test FSM in take off mode with reward for multiple resets
"""


class TestFsm(unittest.TestCase):

    def setUp(self) -> None:
        self.config = {
            'robot_name': 'test_fsm_robot',
            'fsm': True,
            'fsm_mode': 'SingleRun',
            'world_name': 'test_fsm_world',
            'control_mapping': False,
            'waypoint_indicator': False,
        }
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        self.config['output_path'] = self.output_dir

        # spinoff roslaunch
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=self.config,
                                       visible=False)

        self.delay_evaluation = rospy.get_param('/world/delay_evaluation')
        self.state_topic = '/fsm/state'
        self.reward_topic = '/fsm/reward'
        self.reset_topic = '/fsm/reset'
        self.pose_topic = rospy.get_param('/robot/position_sensor/topic')
        self.depth_scan_topic = rospy.get_param('/robot/depth_sensor/topic')
        self.modified_state_topic = rospy.get_param('/robot/modified_state_sensor/topic')
        # subscribe to fsm state and reward (fsm's output)
        subscribe_topics = [
            TopicConfig(topic_name=self.state_topic, msg_type="String"),
            TopicConfig(topic_name=self.reward_topic, msg_type="RosReward"),
        ]

        # create publishers for all relevant sensors < FSM
        publish_topics = [
            TopicConfig(topic_name=self.reset_topic, msg_type='Empty'),
        ]
        for sensor in SensorType:
            if rospy.has_param(f'/robot/{sensor.name}_sensor'):
                publish_topics.append(
                    TopicConfig(topic_name=eval(f'rospy.get_param("/robot/{sensor.name}_sensor/topic")'),
                                msg_type=eval(f'rospy.get_param("/robot/{sensor.name}_sensor/type")'))
                )

        self.ros_topic = TestPublisherSubscriber(
            subscribe_topics=subscribe_topics,
            publish_topics=publish_topics
        )

    def test_occasions(self):
        print('_test_start_up')
        self._test_start_up()
        print('_test_delay_evaluation')
        self._test_delay_evaluation()
        print('_test_step')
        self._test_step()
        print('_test_on_collision')
        self._test_on_collision()
        print('_test_goal_reached')
        self._test_goal_reached()
        print('_test_out_of_time')
        self._test_out_of_time()

    def _test_start_up(self):
        # FSM should start in unknown state, waiting for reset
        # @ startup (before reset)
        safe_wait_till_true('"/fsm/state" in kwargs["ros_topic"].topic_values.keys()',
                            True, 5, 0.1, ros_topic=self.ros_topic)
        self.assertEqual('Unknown', self.ros_topic.topic_values[self.state_topic].data)
        safe_wait_till_true('"/fsm/reward" in kwargs["ros_topic"].topic_values.keys()',
                            True, 5, 0.1, ros_topic=self.ros_topic)
        self.assertEqual(0, self.ros_topic.topic_values[self.reward_topic].reward)
        self.assertEqual('Unknown', self.ros_topic.topic_values[self.reward_topic].termination)

    def _test_delay_evaluation(self):
        time.sleep(0.1)
        self.ros_topic.publishers[self.reset_topic].publish(Empty())
        start_time = rospy.get_time()
        while self.ros_topic.topic_values[self.state_topic].data == 'Unknown':
            rospy.sleep(0.01)
        delay_duration = rospy.get_time() - start_time
        self.assertLess(abs(self.delay_evaluation - delay_duration), 0.2)
        self.assertEqual('Running', self.ros_topic.topic_values[self.state_topic].data)

    def _test_step(self):
        self.ros_topic.publishers[self.pose_topic].publish(get_fake_odometry())
        rospy.sleep(0.1)
        self.ros_topic.publishers[self.pose_topic].publish(get_fake_odometry(1))
        rospy.sleep(0.1)
        self.assertEqual(rospy.get_param('/world/reward/step/termination'),
                         self.ros_topic.topic_values[self.reward_topic].termination)
        reward_distance_travelled = rospy.get_param('/world/reward/step/weights/travelled_distance') * 1
        self.assertEqual(reward_distance_travelled,
                         self.ros_topic.topic_values[self.reward_topic].reward)
        self.ros_topic.publishers[self.pose_topic].publish(get_fake_odometry(1, 1))
        while self.ros_topic.topic_values[self.reward_topic].reward == reward_distance_travelled:
            rospy.sleep(0.1)
        reward_distance_travelled = rospy.get_param('/world/reward/step/weights/travelled_distance') * 2
        self.assertAlmostEqual(reward_distance_travelled,
                               self.ros_topic.topic_values[self.reward_topic].reward,
                               places=1)

    def _test_on_collision(self):
        offset = 3
        self.ros_topic.publishers[self.modified_state_topic].publish(
            get_fake_modified_state([1, 0, 1, 1, offset, 1, 0, 0, 0]))
        time.sleep(0.5)  # make sure modified state is received before collision
        self.ros_topic.publishers[self.depth_scan_topic].publish(get_fake_laser_scan([.2] * 360))
        safe_wait_till_true('kwargs["ros_topic"].topic_values["/fsm/reward"].termination',
                            TerminationType.Failure.name, 4, 0.1, ros_topic=self.ros_topic)
        self.assertAlmostEqual(rospy.get_param('/world/reward/on_collision/weights/distance_between_agents') * offset,
                               self.ros_topic.topic_values[self.reward_topic].reward, places=5)
        self.assertEqual(rospy.get_param('/world/reward/on_collision/termination'),
                         self.ros_topic.topic_values[self.reward_topic].termination)

    def _test_goal_reached(self):
        # reset
        self.ros_topic.publishers[self.reset_topic].publish(Empty())
        time.sleep(0.1)
        safe_wait_till_true('kwargs["ros_topic"].topic_values["/fsm/state"].data',
                            FsmState.Running.name, 3, 0.1, ros_topic=self.ros_topic)
        self.ros_topic.publishers[self.pose_topic].publish(Odometry())
        rospy.sleep(0.1)
        goal_pos = [2, 2, 1]
        self.ros_topic.publishers[self.pose_topic].publish(
            get_fake_odometry(*goal_pos)
        )
        safe_wait_till_true('kwargs["ros_topic"].topic_values["/fsm/state"].data',
                            FsmState.Terminated.name, 3, 0.1, ros_topic=self.ros_topic)
        distance = np.sqrt(sum(np.asarray(goal_pos)**2))
        self.assertEqual(rospy.get_param('/world/reward/goal_reached/weights/distance_from_start') * distance,
                         self.ros_topic.topic_values[self.reward_topic].reward)
        self.assertEqual(rospy.get_param('/world/reward/goal_reached/termination'),
                         self.ros_topic.topic_values[self.reward_topic].termination)

    def _test_out_of_time(self):
        # reset
        self.ros_topic.publishers[self.reset_topic].publish(Empty())
        safe_wait_till_true('kwargs["ros_topic"].topic_values["/fsm/state"].data',
                            FsmState.Running.name, 4, 0.1, ros_topic=self.ros_topic)
        offset = 3
        while self.ros_topic.topic_values[self.reward_topic].termination == 'NotDone':
            rospy.sleep(1)
            self.ros_topic.publishers[self.modified_state_topic].publish(
                get_fake_modified_state([1, 0, 1, 1, offset, 1, 0, 0, 0]))
            rospy.sleep(1)
        iou = 5  # TODO define correct IOU when this is implemented
        self.assertEqual(rospy.get_param('/world/reward/out_of_time/weights/iou') * iou,
                         self.ros_topic.topic_values[self.reward_topic].reward)
        self.assertEqual(rospy.get_param('/world/reward/out_of_time/termination'),
                         self.ros_topic.topic_values[self.reward_topic].termination)

    def tearDown(self) -> None:
        self._ros_process.terminate()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
