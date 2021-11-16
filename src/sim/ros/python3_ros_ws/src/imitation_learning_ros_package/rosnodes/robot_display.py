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
        self._border_width = 400
        self._counter = 0
        self._skip_first_n = 30
        self._skip_every_n = 4
        # Camera matrices
        self._intrinsic_matrix = np.asarray([[100, 0.0, 100], 
                                             [0.0, 100, 100], 
                                             [0.0, 0.0, 1.0]])
        self._cam_to_opt_rotation = R.from_quat([0.5, -0.5, 0.5, 0.5]).as_matrix()
        self._base_to_cam_dict = {
            0: R.from_quat([0, 0, 0, 1]).as_matrix(),
            -13: R.from_quat([0, -0.113, 0, 0.994]).as_matrix(),
            -90: R.from_quat([ 0, 0.7071068, 0, 0.7071068]).as_matrix(),
        }
        self._base_to_cam_translation = np.asarray([0.1, 0, 0])
        self._reference_image_location = None
        self._update_rate_time_tags = []
        self._update_rate = None

        self._logger = get_logger(get_filename_without_extension(__file__), self._output_path)
        self._subscribe()
        self._add_control_specs()
    
    def _subscribe(self):
        # Robot sensors:
        self._view = None
        if rospy.has_param('/robot/camera_sensor'):
            sensor_topic = rospy.get_param('/robot/camera_sensor/topic')
            sensor_type = rospy.get_param('/robot/camera_sensor/type')
            rospy.Subscriber(name=sensor_topic,
                             data_class=eval(sensor_type),
                             callback=self._process_image,
                             callback_args=("observation", {}))
            self._observation_sensor_stats = {'height': 200, 'width': 200, 'depth': 3}
            self._observation = None

        self._mask = None
        self._certainty = None
        rospy.Subscriber(name='/mask', data_class=Image,
                             callback=self._process_image,
                             callback_args=("mask", {}))
        self._mask_sensor_stats = {'height': 200, 'width': 200, 'depth': 1}

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

        # Reward topic
        self._reward = None
        self._terminal_state = TerminationType.Unknown
        rospy.Subscriber(name='/fsm/reward_topic',
                         data_class=RosReward,
                         callback=self._set_field,
                         callback_args=('reward', {}))

        # reference_pose
        self._reference_pose = None
        rospy.Subscriber(name='/reference_pose',
                         data_class=PointStamped,
                         callback=self._set_field,
                         callback_args=('reference_pose', {}))

        # battery state
        self._battery = None
        rospy.Subscriber(name='/bebop/states/common/CommonState/BatteryStateChanged',
                         data_class=CommonCommonStateBatteryStateChanged,
                         callback=self._set_field,
                         callback_args=('battery', {}))

        # trajectory
        self._trajectory = []
        if rospy.has_param('/robot/position_sensor'):
            rospy.Subscriber(name=rospy.get_param('/robot/position_sensor/topic'),
                            data_class=eval(rospy.get_param('/robot/position_sensor/type')),
                            callback=self._set_field,
                            callback_args=('trajectory', {})
            )

        # camera orientation
        self._camera_orientation = None
        rospy.Subscriber(name='/bebop/camera_control',
                        data_class=Twist,
                        callback=self._set_field,
                        callback_args=('camera_orientation', {}))

    def _reset(self, msg: Empty = None):
        self._reward = None
        self._fsm_state = None
        self._terminal_state = None
        self._action = None
        self._reference_image_location = None

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
        if rospy.has_param('/actor/keyboard/specs/message'):
            self._control_specs['message'] = rospy.get_param('/actor/keyboard/specs/message')
            print(self._control_specs['message'])
    
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
            msg = '[' + ', '.join(f'{e:.1f}' for e in self._action.value) + ']'
            image = cv2.putText(image, msg, (3, origin[1] + 55 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (1, 0, 0),
                                thickness=2)
        return image

    def _draw_top_down_waypoint(self, image: np.ndarray, height: int = -1) -> np.ndarray:
        if self._reference_pose is not None:
            origin = (self._border_width - 50, int(image.shape[0] / 2) if height == -1 else height + 50)
            scale = 60
            try:
                reference_point = (int(origin[0] - scale * self._reference_pose[1]),
                                   int(origin[1] - scale * self._reference_pose[0]))
            except ValueError:
                reference_point = (int(origin[0]), int(origin[1]))
            image = cv2.circle(image, origin, radius=4, color=(0, 0, 0, 0.3), thickness=2)
            image = cv2.arrowedLine(image, origin, reference_point, (1, 0, 0), thickness=2)
        return image
    
    def _write_info(self, image: np.ndarray, height: int = 0) -> Tuple[np.ndarray, int]:
        for key, msg in {'fsm': self._fsm_state.name if self._fsm_state is not None else None,
                         'wp': 'wp: '+' '.join(f'{e:.3f}' for e in self._reference_pose)
                         if self._reference_pose is not None else None,
                         'reward': f'reward: {self._reward:.2f}' if self._reward is not None else None,
                         'battery': f'battery: {self._battery}%' if self._battery is not None else None,
                         'camera': f'camera pitch: {self._camera_orientation:.0f}' if self._camera_orientation is not None else None,
                         'certainty': f'certainty {self._certainty:.3f}' if self._certainty is not None else None,
                         'rate': f'update rate: {self._update_rate:.3f} fps' if self._update_rate is not None else None}.items():
            if msg is not None:
                image = cv2.putText(image, msg, (3, height + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (1, 0, 0), thickness=2)
                height += 15
        return image, height

    def _draw(self):
        self._counter += 1
        if self._counter < self._skip_first_n or self._counter % self._skip_every_n == 0:
            return
        image = np.zeros((200, 400, 3))
        result = self._process_mask()
        if result is not None:
            image[:, 200:, :] = result
        result = self._process_observation()
        if result is not None:
            image[:, :200, :] = result
        border = np.ones((image.shape[0], self._border_width, 3), dtype=image.dtype)
        image = np.concatenate([border, image, border], axis=1)
        image, height = self._write_info(image)
        image = self._draw_action(image, height)
        image = self._draw_top_down_waypoint(image, height)
        image = self._write_control_specs(image)
        cv2.imshow("Image window", image)
        cv2.waitKey(1)        

    def _process_image(self, msg: Union[Image, CompressedImage], args: tuple) -> None:
        field_name, sensor_stats = args
        if field_name == "observation":
            self._view = msg
        elif field_name == "mask":
            self._mask = msg
            self._update_rate_time_tags.append(time.time_ns())
            if len(self._update_rate_time_tags) >= 5:
                differences = [
                    (self._update_rate_time_tags[i+1] - self._update_rate_time_tags[i]) * 10**-9 
                    for i in range(len(self._update_rate_time_tags) - 1)]
                self._update_rate = 1/(np.mean(differences))
                self._update_rate_time_tags.pop(0)

    def _process_observation(self):
        if self._view is None:
            return None
        msg = copy.deepcopy(self._view)
        self._view = None
        image = process_image(msg, self._observation_sensor_stats) if isinstance(msg, Image) else process_compressed_image(msg, self._observation_sensor_stats)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def _process_mask(self):
        if self._mask is None:
            return None
        msg = copy.deepcopy(self._mask)
        self._mask = None
        image = process_image(msg, self._mask_sensor_stats) if isinstance(msg, Image) else process_compressed_image(msg, self._mask_sensor_stats)        
        self._certainty = np.mean(image[::10, ::10])
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
        elif field_name == 'reward':
            self._reward = msg.reward
            self._terminal_state = TerminationType[msg.termination]
        elif field_name == 'reference_pose':
            self._reference_pose = np.asarray([msg.point.x, msg.point.y, msg.point.z])
            self._calculate_ref_in_img()
        elif field_name == 'battery':
            self._battery = msg.percent
        elif field_name == 'trajectory':
            global_pose = process_odometry(msg)
            self._trajectory.append(global_pose)
        elif field_name == 'camera_orientation':
            self._camera_orientation = float(msg.angular.y)
        else:
            raise NotImplementedError

    def _calculate_ref_in_img(self):
        if self._reference_pose is None:
            return None
        #print(f'reference pose: {self._reference_pose}')
        # translate to camera location
        p = self._reference_pose - self._base_to_cam_translation
        # rotate to camera orientation
        base_to_cam_rot = self._base_to_cam_dict[self._camera_orientation if self._camera_orientation is not None else 0]
        p = np.matmul(base_to_cam_rot, p)
        #print(f'reference in camera frame: {p}')
        # rotate to optical frame
        p = np.matmul(self._cam_to_opt_rotation, p)
        #print(f'reference in optical frame: {p}')
        # map to image coords
        p = np.matmul(self._intrinsic_matrix, p)
        self._reference_image_location = p / p[2]
        #print(f'pixel coordinates: {self._reference_image_location}')

    def _write_control_specs(self, image: np.ndarray) -> np.ndarray:
        height = 0
        for key, value in self._control_specs.items():
            if key != 'message':
                msg = f'{key}: {value}'
                image = cv2.putText(image, msg, (image.shape[1] - self._border_width + 5, height + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (1, 0, 0), thickness=2)
            else:
                step = 40
                for index in range(0, len(value), step):
                    line = value[index:min([len(value), index + step])]
                    image = cv2.putText(image, line, (image.shape[1] - self._border_width + 5, height + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (1, 0, 0), thickness=2)
                    height += 15
            height += 15
        return image

    def _cleanup(self):
        cv2.destroyAllWindows()

    def run(self):
        rate = rospy.Rate(self._rate_fps)
        while not rospy.is_shutdown():
            self._draw()
            rate.sleep()
        self._cleanup()


if __name__ == "__main__":
    robot_display = RobotDisplay()
    robot_display.run()
