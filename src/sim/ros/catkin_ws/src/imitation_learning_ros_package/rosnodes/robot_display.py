#!/usr/bin/python3.8
import time

import rospy
import numpy as np
from sensor_msgs.msg import Image

from src.core.logger import get_logger, cprint
from src.sim.ros.src.utils import process_image, get_output_path
from src.core.utils import camelcase_to_snake_format, get_filename_without_extension


class RobotDisplay:

    def __init__(self):
        rospy.init_node('robot_display')
        stime = time.time()
        max_duration = 60
        while not rospy.has_param('/output_path') and time.time() < stime + max_duration:
            time.sleep(0.01)
        self._output_path = get_output_path()
        self._rate_fps = 10
        self._logger = get_logger(get_filename_without_extension(__file__), self._output_path)
        self._subscribe()

    def _subscribe(self):
        # Robot sensors:
        sensor = '/robot/forward_camera'
        if rospy.has_param(f'{sensor}_topic'):
            sensor_topic = rospy.get_param(f'{sensor}_topic')
            sensor_type = rospy.get_param(f'{sensor}_type')
            sensor_callback = f'_process_{camelcase_to_snake_format(sensor_type)}'
            if sensor_callback not in self.__dir__():
                cprint(f'Could not find sensor_callback {sensor_callback}', self._logger)
            sensor_stats = rospy.get_param(f'{sensor}_stats') if rospy.has_param(f'{sensor}_stats') else {}
            rospy.Subscriber(name=sensor_topic,
                             data_class=eval(sensor_type),
                             callback=eval(f'self.{sensor_callback}'),
                             callback_args=(sensor_topic, sensor_stats))

    def _process_image(self, msg: Image, args: tuple) -> None:
        sensor_topic, sensor_stats = args
        processed_image = process_image(msg, sensor_stats)

    def run(self):
        rate = rospy.Rate(self._rate_fps)
        while not rospy.is_shutdown():
            rate.sleep()


if __name__ == "__main__":
    robot_display = RobotDisplay()
    robot_display.run()
