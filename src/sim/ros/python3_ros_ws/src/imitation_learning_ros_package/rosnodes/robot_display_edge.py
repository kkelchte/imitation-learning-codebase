#!/usr/bin/python3.8
import time
from typing import Union, Tuple
import copy
import rospy
import cv2
from scipy.spatial.transform import Rotation as R
import numpy as np
from bebop_msgs.msg import CommonCommonStateBatteryStateChanged
from imitation_learning_ros_package.msg import RosReward
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Empty
from src.core.data_types import Action, TerminationType
from src.core.logger import get_logger, cprint, MessageType
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.src.utils import process_image, process_compressed_image, get_output_path, process_odometry, process_twist, transform
from src.core.utils import camelcase_to_snake_format, get_filename_without_extension


class EdgeDisplay:

    def __init__(self):
        rospy.init_node('edge_display')
        stime = time.time()
        max_duration = 120
        while not rospy.has_param('/output_path') and time.time() < stime + max_duration:
            time.sleep(0.01)
        self._output_path = get_output_path()
        self._rate_fps = 10
        self._border_width = 400
        self._counter = 0
        self._skip_first_n = 30
        self._skip_every_n = 4
        self._subscribe()
        self._fsm_state_name_mapper = {
            "Unknown": "Loading",
            "Running": "Autonomous",
            "TakenOver": "Pilot",
            "Terminated": "Finished",
        }
    
    def _subscribe(self):
        # Robot sensors:
        self._view = None
        self._view_msg = None
        if rospy.has_param('/robot/camera_sensor'):
            sensor_topic = '/bebop/image_raw/compressed'
            sensor_type = 'CompressedImage'
            rospy.Subscriber(name=sensor_topic,
                             data_class=eval(sensor_type),
                             callback=self._process_image,
                             callback_args=("observation", {}))
            self._observation = None
            self._observation_sensor_stats = {'height': 400, 'width': 400, 'depth': 3}

        self._mask = None
        self._mask_msg = None
        rospy.Subscriber(name='/mask', data_class=Image,
                             callback=self._process_image,
                             callback_args=("mask", {}))
        self._mask_sensor_stats = {'height': 400, 'width': 400, 'depth': 1}

        rospy.Subscriber('/fsm/reset', Empty, self._reset)

        # Applied action
        self._action = None
        if rospy.has_param('/robot/command_topic'):
            rospy.Subscriber(name=rospy.get_param('/robot/command_topic'),
                             data_class=Twist,
                             callback=self._set_field,
                             callback_args=('action', {}))
        # fsm state
        self._fsm_state = None
        rospy.Subscriber(name='/fsm/state',
                         data_class=String,
                         callback=self._set_field,
                         callback_args=('fsm_state', {}))

        # reference_pose
        self._reference_pose = None
        rospy.Subscriber(name='/reference_pose',
                         data_class=PointStamped,
                         callback=self._set_field,
                         callback_args=('reference_pose', {}))

    def _reset(self, msg: Empty = None):
        self._fsm_state = None
        self._action = None
        self._reference_image_location = None

    def _draw_action(self, image: np.ndarray, height: int = -1) -> np.ndarray:
        if self._action is not None:
            forward_speed = 200 * self._action.value[0]
            direction = np.arccos(self._action.value[-1])
            origin = (50, int(image.shape[0] / 2) if height == -1 else height + 50)
            try:
                steering_point = (int(origin[0] - forward_speed * np.cos(direction)),
                                  int(origin[1] - forward_speed * np.sin(direction)))
            except ValueError:
                steering_point = (int(origin[0]), int(origin[1]))
            image = cv2.circle(image, origin, radius=20, color=(0, 0, 0, 0.3), thickness=3)
            image = cv2.arrowedLine(image, origin, steering_point, (255, 0, 0), thickness=1)
            #msg = '[' + ', '.join(f'{e:.1f}' for e in self._action.value) + ']'
            msg = 'velocity'
            image = cv2.putText(image, msg, (3, origin[1] + 55 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (1, 0, 0),
                                thickness=2)
        return image
    
    def _write_info(self, image: np.ndarray, height: int = 0) -> Tuple[np.ndarray, int]:
        for key, msg in {'state': self._fsm_state_name_mapper[self._fsm_state.name] if self._fsm_state is not None else None,
                         'wp': 'waypoint: '+' '.join(f'{e:.3f}' for e in self._reference_pose)
                         if self._reference_pose is not None else None}.items():
            if msg is not None:
                if key == 'state' and self._fsm_state.name == 'Running':
                    image = cv2.putText(image, msg, (3, height + 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 1, 0), thickness=2)
                elif key == 'state' and self._fsm_state.name == 'TakenOver':
                    image = cv2.putText(image, msg, (3, height + 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 1), thickness=2)
                else:
                    image = cv2.putText(image, msg, (3, height + 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (1, 0, 0), thickness=2)
                height += 15
        return image, height

    def _draw(self):
        self._counter += 1
        if self._counter < self._skip_first_n or self._counter % self._skip_every_n == 0:
            return
        image = np.zeros((400, 800, 3))
        result = self._process_mask()
        self._mask = result if result is not None else self._mask
        if self._mask is not None:
            image[:, 400:, :] = self._mask
        result = self._process_observation()
        self._view = result if result is not None else self._view
        if self._view is not None:
            image[:, :400, :] = self._view
        border = np.ones((image.shape[0], self._border_width, 3), dtype=image.dtype)
        image = np.concatenate([border, image], axis=1)
        image, height = self._write_info(image)
        image = self._draw_action(image, height + 50)
        cv2.imshow("Edge Display", image)
        cv2.waitKey(1)        

    def _process_image(self, msg: Union[Image, CompressedImage], args: tuple) -> None:
        field_name, _ = args
        if field_name == "observation":
            self._view_msg = msg
        elif field_name == "mask":
            self._mask_msg = msg

    def _process_observation(self):
        if self._view_msg is None:
            return None
        msg = copy.deepcopy(self._view_msg)
        self._view_msg = None
        image = process_image(msg, self._observation_sensor_stats) if isinstance(msg, Image) else process_compressed_image(msg, self._observation_sensor_stats)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if isinstance(msg, Image) else image

    def _process_mask(self):
        if self._mask_msg is None:
            return None
        msg = copy.deepcopy(self._mask_msg)
        self._mask_msg = None
        image = process_image(msg, self._mask_sensor_stats) if isinstance(msg, Image) else process_compressed_image(msg, self._mask_sensor_stats)        
        image -= image.min()
        image /= image.max()
        image *= 255
        return cv2.applyColorMap(image.astype(np.uint8), cv2.COLORMAP_AUTUMN)/255.

    def _set_field(self, msg: Union[String, Twist, RosReward, CommonCommonStateBatteryStateChanged, Odometry],
                   args: Tuple) -> None:
        field_name, _ = args
        if field_name == 'fsm_state':
            self._fsm_state = FsmState[msg.data]
        elif field_name == 'action':
            self._action = Action(actor_name='applied_action',
                                  value=process_twist(msg).value)
        elif field_name == 'reference_pose':
            self._reference_pose = np.asarray([msg.point.x, msg.point.y, msg.point.z])
        else:
            raise NotImplementedError

    def _cleanup(self):
        cv2.destroyAllWindows()

    def run(self):
        rate = rospy.Rate(self._rate_fps)
        while not rospy.is_shutdown():
            self._draw()
            rate.sleep()
        self._cleanup()


if __name__ == "__main__":
    robot_display = EdgeDisplay()
    robot_display.run()
