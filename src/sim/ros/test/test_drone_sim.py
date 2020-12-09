#!/usr/bin/python3.8
import os
import shutil
import unittest
import time

import rospy

from src.core.utils import count_grep_name, get_data_dir
from src.core.utils import get_filename_without_extension
from src.core.data_types import ProcessState
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.test.common_utils import TopicConfig, TestPublisherSubscriber
from std_srvs.srv import Empty as Emptyservice, EmptyRequest


class TestRos(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        config = {
            'random_seed': 123,
            'gazebo': 'true',
            'world_name': 'empty',
            'robot_name': 'drone_sim',
            'output_path': self.output_dir
        }
        self.ros_process = RosWrapper(launch_file='load_ros.launch',
                                      config=config,
                                      visible=False)
        subscribe_topics = [TopicConfig(topic_name=rospy.get_param(f'/robot/{sensor}_sensor/topic'),
                                        msg_type=rospy.get_param(f'/robot/{sensor}_sensor/type'))
                            for sensor in ['camera', 'position', 'depth']]
        self.ros_topic = TestPublisherSubscriber(
            subscribe_topics=subscribe_topics,
            publish_topics=[]
        )
        self._unpause_client = rospy.ServiceProxy('/gazebo/unpause_physics', Emptyservice)

    def test_drone_sim(self):
        self.assertEqual(self.ros_process.get_state(), ProcessState.Running)
        self._unpause_client.wait_for_service()
        self._unpause_client(EmptyRequest())
        time.sleep(1)
        for sensor in ['camera', 'position', 'depth']:  # collision < wrench, only published when turned upside down
            self.assertTrue(rospy.get_param(f'/robot/{sensor}_sensor/topic') in self.ros_topic.topic_values.keys())

    def tearDown(self) -> None:
        self.ros_process.terminate()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
