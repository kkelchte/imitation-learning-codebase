#!/usr/bin/python3.8
import time
from typing import Union, Tuple

import rospy
import cv2
import numpy as np
from imitation_learning_ros_package.msg import RosReward
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Empty, Float32MultiArray

from src.core.data_types import Action, TerminationType
from src.core.logger import get_logger, cprint, MessageType
from src.sim.ros.catkin_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.src.utils import process_image, get_output_path, adapt_twist_to_action
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
        self._counter = 0
        self._skip_first_n = 30
        self._skip_every_n = 4
        self._logger = get_logger(get_filename_without_extension(__file__), self._output_path)
        self._subscribe()

    def _draw_action(self, image: np.ndarray) -> np.ndarray:
        if self._action is not None:
            forward_speed = 200 * self._action.value[0]
            direction = np.arccos(self._action.value[-1])
            origin = (int(image.shape[1]/2), int(image.shape[0]/2))
            steering_point = (int(origin[0] - forward_speed * np.cos(direction)),
                              int(origin[1] - forward_speed * np.sin(direction)))
            image = cv2.line(image, origin, steering_point, (255, 0, 0), thickness=1)
        return image

    def _draw_fsm_state(self, image: np.ndarray) -> np.ndarray:
        if self._fsm_state is not None:
            image = cv2.putText(image, self._fsm_state.name, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0),
                                thickness=1)
        return image

    def _draw_waypoint(self, image: np.ndarray) -> np.ndarray:
        if self._waypoint is not None:
            image = cv2.putText(image, str(self._waypoint), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0),
                                thickness=1)
        return image

    def _draw_reward(self, image: np.ndarray) -> np.ndarray:
        if self._reward is not None:
            image = cv2.putText(image, str(self._reward), (3, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0),
                                thickness=2)
        return image

    def _draw(self, image: np.ndarray):
        self._counter += 1
        if self._counter < self._skip_first_n or self._counter % self._skip_every_n == 0:
            return

        image = self._draw_action(image)
        image = self._draw_fsm_state(image)
        image = self._draw_waypoint(image)
        cv2.imshow("Image window", image)
        cv2.waitKey(3)

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
        rospy.Subscriber(rospy.get_param('/fsm/reset_topic', '/reset'), Empty, self._reset)

        self._action = None
        if rospy.has_param('/robot/command_topic'):
            rospy.Subscriber(name=rospy.get_param('/robot/command_topic'),
                             data_class=Twist,
                             callback=self._set_field,
                             callback_args=('action', {}))
        # fsm state
        self._fsm_state = None
        rospy.Subscriber(name=rospy.get_param('/fsm/state_topic'),
                         data_class=String,
                         callback=self._set_field,
                         callback_args=('fsm_state', {}))

        # Reward topic
        self._reward = None
        self._terminal_state = TerminationType.Unknown
        rospy.Subscriber(name=rospy.get_param('/fsm/reward_topic', ''),
                         data_class=RosReward,
                         callback=self._set_field,
                         callback_args=('reward', {}))

        # waypoint
        self._waypoint = None
        rospy.Subscriber(name='/waypoint_indicator/current_waypoint',
                         data_class=Float32MultiArray,
                         callback=self._set_field,
                         callback_args=('waypoint', {}))

    def _reset(self, msg: Empty = None):
        self._reward = None
        self._fsm_state = None
        self._terminal_state = None
        self._action = None

    def _process_image(self, msg: Image, args: tuple) -> None:
        sensor_topic, sensor_stats = args
        image = process_image(msg, sensor_stats)
        resized_image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)
        self._draw(resized_image)

    def _set_field(self, msg: Union[String, Twist, RosReward], args: Tuple) -> None:
        field_name, sensor_stats = args
        if field_name == 'fsm_state':
            self._fsm_state = FsmState[msg.data]
        elif field_name == 'action':
            self._action = Action(actor_name='applied_action',
                                  value=adapt_twist_to_action(msg).value)
        elif field_name == 'reward':
            self._reward = msg.reward
            self._terminal_state = TerminationType[msg.termination]
        elif field_name == 'waypoint':
            self._waypoint = np.asarray(msg.data)
        else:
            raise NotImplementedError
        cprint(f'set field {field_name}', self._logger, msg_type=MessageType.debug)

    def _cleanup(self):
        cv2.destroyAllWindows()

    def run(self):
        rate = rospy.Rate(self._rate_fps)
        while not rospy.is_shutdown():
            rate.sleep()
        self._cleanup()


if __name__ == "__main__":
    robot_display = RobotDisplay()
    robot_display.run()
