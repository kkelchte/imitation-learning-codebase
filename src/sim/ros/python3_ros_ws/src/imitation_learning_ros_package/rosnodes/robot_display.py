#!/usr/bin/python3.8
import time
from typing import Union, Tuple

import rospy
import cv2
import numpy as np
from bebop_msgs.msg import CommonCommonStateBatteryStateChanged
from imitation_learning_ros_package.msg import RosReward
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Empty, Float32MultiArray

from src.core.data_types import Action, TerminationType
from src.core.logger import get_logger, cprint, MessageType
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.src.utils import process_image, get_output_path, adapt_twist_to_action
from src.core.utils import camelcase_to_snake_format, get_filename_without_extension

JOYSTICK_BUTTON_MAPPING = {
    0: 'SQUARE',
    1: 'X',
    2: 'O',
    3: 'TRIANGLE',
    4: 'L1',
    5: 'R1',
    7: 'R2',
    9: 'start'
}


class RobotDisplay:

    def __init__(self):
        rospy.init_node('robot_display')
        stime = time.time()
        max_duration = 60
        while not rospy.has_param('/output_path') and time.time() < stime + max_duration:
            time.sleep(0.01)
        self._output_path = get_output_path()
        self._rate_fps = 10
        self._border_width = 300
        self._counter = 0
        self._skip_first_n = 30
        self._skip_every_n = 4
        self._logger = get_logger(get_filename_without_extension(__file__), self._output_path)
        self._subscribe()
        self._add_control_specs()

    def _add_control_specs(self):
        self._control_specs = {}
        if rospy.has_param('/actor/joystick/teleop'):
            specs = rospy.get_param('/actor/joystick/teleop')
            for name in ['takeoff', 'land', 'emergency', 'flattrim', 'go', 'overtake', 'toggle_camera_forward_down']:
                if name in specs.keys():
                    button_integers = specs[name][
                        'deadman_buttons' if 'deadman_buttons' in specs[name].keys() else 'buttons']
                    self._control_specs[name] = ' '.join([JOYSTICK_BUTTON_MAPPING[button_integer]
                                                          for button_integer in button_integers])

    def _draw_action(self, image: np.ndarray, height: int = -1) -> np.ndarray:
        if self._action is not None:
            forward_speed = 200 * self._action.value[0]
            direction = np.arccos(self._action.value[-1])
#            origin = (int(image.shape[1] / 2), int(image.shape[0] / 2))
            origin = (50, int(image.shape[0] / 2) if height == -1 else height + 50)
            steering_point = (int(origin[0] - forward_speed * np.cos(direction)),
                              int(origin[1] - forward_speed * np.sin(direction)))
            image = cv2.circle(image, origin, radius=20, color=(0, 0, 0, 0.3), thickness=3)
            image = cv2.arrowedLine(image, origin, steering_point, (255, 0, 0), thickness=1)
            msg = '[' + ', '.join(f'{e:.1f}' for e in self._action.value) + ']'
            image = cv2.putText(image, msg, (3, origin[1] + 55 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (1, 0, 0),
                                thickness=2)
        return image

    def _write_info(self, image: np.ndarray, height: int = 0) -> Tuple[np.ndarray, int]:
        for key, msg in {'fsm': self._fsm_state.name if self._fsm_state is not None else None,
                         'wp': 'wp: '+' '.join(f'{e:.3f}' for e in self._waypoint)
                         if self._waypoint is not None else None,
                         'reward': f'reward: {self._reward:.2f}' if self._reward is not None else None,
                         'battery': f'battery: {self._battery}%' if self._battery is not None else None}.items():
            if msg is not None:
                image = cv2.putText(image, msg, (3, height + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (1, 0, 0), thickness=2)
                height += 15
        return image, height

    def _write_control_specs(self, image: np.ndarray) -> np.ndarray:
        height = 0
        for key, value in self._control_specs.items():
            msg = f'{key}: {value}'
            image = cv2.putText(image, msg, (image.shape[1] - self._border_width + 5, height + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (1, 0, 0), thickness=2)
            height += 15
        return image

    def _draw(self, image: np.ndarray):
        self._counter += 1
        if self._counter < self._skip_first_n or self._counter % self._skip_every_n == 0:
            return
        border = np.ones((image.shape[0], self._border_width, 3), dtype=image.dtype)
        image = np.concatenate([border, image, border], axis=1)
        image, height = self._write_info(image)
        image = self._draw_action(image, height)
        image = self._write_control_specs(image)
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

        # Applied action
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

        # battery state
        self._battery = None
        rospy.Subscriber(name='/bebop/states/common/CommonState/BatteryStateChanged',
                         data_class=CommonCommonStateBatteryStateChanged,
                         callback=self._set_field,
                         callback_args=('battery', {}))

    def _reset(self, msg: Empty = None):
        self._reward = None
        self._fsm_state = None
        self._terminal_state = None
        self._action = None

    def _process_image(self, msg: Image, args: tuple) -> None:
        sensor_topic, sensor_stats = args
        image = process_image(msg, sensor_stats)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if sum(image.shape) < 3000:
            image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)
        self._draw(image)

    def _set_field(self, msg: Union[String, Twist, RosReward, CommonCommonStateBatteryStateChanged],
                   args: Tuple) -> None:
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
        elif field_name == 'battery':
            self._battery = msg.percent
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
