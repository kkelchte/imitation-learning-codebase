from typing import List

from dataclasses import dataclass

import rospy
from nav_msgs.msg import *  # Do not remove!
from std_msgs.msg import *  # Do not remove!
from sensor_msgs.msg import *  # Do not remove!
from geometry_msgs.msg import *  # Do not remove!


@dataclass
class TopicConfig:
    topic_name: str
    msg_type: str


class TestPublisherSubscriber:

    def __init__(self, subscribe_topics: List[TopicConfig], publish_topics: List[TopicConfig]):
        self.topic_values = {}
        self._subscribe(subscribe_topics)
        self._set_publishers(publish_topics)
        rospy.init_node(f'test_fsm', anonymous=True)

    def _subscribe(self, subscribe_topics: List[TopicConfig]):
        for topic_config in subscribe_topics:
            rospy.Subscriber(topic_config.topic_name,
                             eval(topic_config.msg_type),
                             self._store,
                             callback_args=topic_config.topic_name)

    def _set_publishers(self, publish_topics: List[TopicConfig]):
        self.publishers = {}
        for topic_config in publish_topics:
            self.publishers[topic_config.topic_name] = rospy.Publisher(topic_config.topic_name,
                                                                       eval(topic_config.msg_type),
                                                                       queue_size=10)

    def _store(self, msg, topic_name: str):
        self.topic_values[topic_name] = msg if not hasattr(msg, 'data') else msg.data
