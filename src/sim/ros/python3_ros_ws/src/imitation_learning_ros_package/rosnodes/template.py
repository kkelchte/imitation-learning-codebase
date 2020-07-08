#!/usr/bin/python3.8

""" Reason

"""
import os
import time

import rospy

from src.core.logger import get_logger, cprint
from src.sim.ros.src.utils import get_output_path


class ClassName:

    def __init__(self):
        start_time = time.time()
        max_duration = 60
        while not rospy.has_param('/output_path') and time.time() < start_time + max_duration:
            time.sleep(0.1)
        self._output_path = get_output_path()
        self._logger = get_logger(os.path.basename(__file__), self._output_path)
        self._rate_fps = rospy.Rate(100)
        rospy.init_node('class_name')

    def run(self):
        cprint(f'started with rate {self._rate_fps}', self._logger)
        while not rospy.is_shutdown():
            self._rate_fps.sleep()


if __name__ == "__main__":
    instant_name = ClassName()
    instant_name.run()
