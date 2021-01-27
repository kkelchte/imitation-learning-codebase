#!/usr/bin/python3.8
"""
Modifies sensor information to a state or observation used by rosenvironment.
Defines separate modes:
Default mode:
    combine tf topics of tracking and fleeing agent in global frame
"""
import time

import numpy as np
import rospy
from sensor_msgs.msg import Image

from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension, camelcase_to_snake_format, ros_message_to_type_str
from imitation_learning_ros_package.msg import CombinedGlobalPoses
from src.sim.ros.src.utils import process_odometry, process_pose_stamped, euler_from_quaternion, get_output_path, \
    process_combined_global_poses, array_to_combined_global_pose, calculate_bounding_box


class ModifiedStateFrameVisualizer:

    def __init__(self):
        stime = time.time()
        max_duration = 60
        while not rospy.has_param('/modified_state_publisher/mode') and time.time() < stime + max_duration:
            time.sleep(0.01)

        self._output_path = get_output_path()
        self._logger = get_logger(get_filename_without_extension(__file__), self._output_path)

        rospy.Subscriber(rospy.get_param('/robot/modified_state_sensor/topic', '/modified_state'),
                         eval(rospy.get_param('/robot/modified_state_sensor/type', 'CombinedGlobalPoses')),
                         self._process_state_and_publish_frame)
        self._publisher = rospy.Publisher('/modified_state_frame',
                                          Image, queue_size=10)
        cprint(f"subscribe to {rospy.get_param('/robot/modified_state_sensor/topic', '/modified_state')}", self._logger)
        rospy.init_node('modified_state_frame_visualizer')

    def _publish_combined_global_poses(self, data: np.ndarray) -> None:
        resolution = (100, 100)
        pos0, w0, h0, pos1, w1, h1 = calculate_bounding_box(state=data,
                                                            resolution=resolution)
        frame = np.zeros(resolution)
        frame[pos0[0]:pos0[0] + w0, pos0[1]:pos0[1] + h0] = 255
        frame[pos1[0]:pos1[0] + w1, pos1[1]:pos1[1] + h1] = 125

        image = Image()
        image.data = frame.astype(np.uint8).flatten().tolist()
        image.height = resolution[0]
        image.width = resolution[1]
        image.encoding = 'mono8'
        self._publisher.publish(image)

    def _process_state_and_publish_frame(self, msg: CombinedGlobalPoses):
        msg_type = camelcase_to_snake_format(ros_message_to_type_str(msg))
        data = eval(f'process_{msg_type}(msg)')
        cprint(f'received message {data} of type {msg_type}')
        if msg_type == 'combined_global_poses':
            self._publish_combined_global_poses(data)

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()


if __name__ == "__main__":
    publisher = ModifiedStateFrameVisualizer()
    publisher.run()
