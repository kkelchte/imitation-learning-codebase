#!/usr/bin/python3.7

""" Reason

"""
import time

import rospy

from src.core.logger import get_logger, cprint


class ClassName:

    def __init__(self):
        while not rospy.has_param('output_path'):
            time.sleep(0.1)
        self._logger = get_logger('class_name', rospy.get_param('output_path'))
        cprint('started', self._logger)

        rospy.init_node('class_name')


def run():
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        rate.sleep()


if __name__ == "__main__":
    instant_name = ClassName()
    run()
