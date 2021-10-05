from typing import List, Iterable, Sequence

from dataclasses import dataclass
import numpy as np
import rospy
from nav_msgs.msg import *  # Do not remove!
from std_msgs.msg import *  # Do not remove!
from sensor_msgs.msg import *  # Do not remove!
from geometry_msgs.msg import *  # Do not remove!
from bebop_msgs.msg import CommonCommonStateBatteryStateChanged

from imitation_learning_ros_package.msg import *

from src.sim.ros.src.utils import process_odometry


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
        self.last_received_sensor_readings = []

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
        self.topic_values[topic_name] = msg


def compare_odometry(first_msg: Odometry, second_msg: Odometry) -> bool:
    first_odom = process_odometry(first_msg)
    second_odom = process_odometry(second_msg)
    return sum(first_odom - second_odom) < 0.1


def get_fake_image():
    image = Image()
    image.data = [int(5)]*300*600*3
    image.height = 600
    image.width = 300
    image.encoding = 'rgb8'
    return image


def get_fake_modified_state(data: Sequence = np.zeros((9,))) -> CombinedGlobalPoses:
    msg = CombinedGlobalPoses()
    msg.tracking_x = data[0]
    msg.tracking_y = data[1]
    msg.tracking_z = data[2]
    msg.fleeing_x = data[3]
    msg.fleeing_y = data[4]
    msg.fleeing_z = data[5]
    msg.tracking_roll = data[6]
    msg.tracking_pitch = data[7]
    msg.tracking_yaw = data[8]
    return msg


def get_fake_laser_scan(ranges=None):
    scan = LaserScan()
    scan.ranges = [1.5]*360 if ranges is None else ranges
    return scan


def get_fake_odometry(x: float = 0, y: float = 0, z: float = 0,
                      xq: float = 0, yq: float = 0, zq: float = 0, wq: float = 1):
    odometry = Odometry()
    odometry.pose.pose.position.x = x
    odometry.pose.pose.position.y = y
    odometry.pose.pose.position.z = z
    odometry.pose.pose.orientation.x = xq
    odometry.pose.pose.orientation.y = yq
    odometry.pose.pose.orientation.z = zq
    odometry.pose.pose.orientation.w = wq
    return odometry


def get_fake_pose_stamped(x: float = 0, y: float = 0, z: float = 0,
                          xq: float = 0, yq: float = 0, zq: float = 0, wq: float = 1):
    pose_stamped = PoseStamped()
    pose_stamped.pose.position.x = x
    pose_stamped.pose.position.y = y
    pose_stamped.pose.position.z = z
    pose_stamped.pose.orientation.x = xq
    pose_stamped.pose.orientation.y = yq
    pose_stamped.pose.orientation.z = zq
    pose_stamped.pose.orientation.w = wq
    return pose_stamped


def get_fake_combined_global_poses(tx: float = 0,
                                   ty: float = 0,
                                   tz: float = 0,
                                   fx: float = 0,
                                   fy: float = 0,
                                   fz: float = 0,
                                   troll: float = 0,
                                   tpitch: float = 0,
                                   tyaw: float = 0) -> CombinedGlobalPoses:
    msg = CombinedGlobalPoses()
    msg.tracking_x = tx
    msg.tracking_y = ty
    msg.tracking_z = tz
    msg.fleeing_x = fx
    msg.fleeing_y = fy
    msg.fleeing_z = fz
    msg.tracking_roll = troll
    msg.tracking_pitch = tpitch
    msg.tracking_yaw = tyaw
    return msg

def get_random_twist():
    msg = Twist()
    msg.linear.x = np.random.uniform(-1, 1)
    msg.linear.y = np.random.uniform(-1, 1)
    msg.linear.z = np.random.uniform(-1, 1)
    msg.angular.x = 0
    msg.angular.y = 0
    msg.angular.z = np.random.uniform(-1, 1)
    return msg


def get_random_image(size: tuple = (200, 200, 3)):
    image = Image()
    image.data = list(np.random.randint(0, 255, size=size[0] * size[1] * size[2], dtype=np.uint8))
    image.height = size[0]
    image.width = size[1]
    image.encoding = 'rgb8' if size[2] == 3 else 'mono8'
    return image


def get_random_reference_pose():
    msg = PointStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = 'agent'
    msg.point.x = np.random.uniform(1, 3)
    msg.point.y = np.random.uniform(-3, 3)
    msg.point.z = np.random.uniform(0.5, 3)
    return msg
