#!/usr/data/kkelchtel/miniconda3/envs/venv/bin/python3.8
from dataclasses import dataclass
from functools import update_wrapper
import time
from typing import Union, Tuple
from tkinter import Tk, Label, Canvas, Frame
from copy import deepcopy

import rospy
import cv2
from cv_bridge import CvBridge
import numpy as np
from bebop_msgs.msg import CommonCommonStateBatteryStateChanged
from imitation_learning_ros_package.msg import RosReward
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Empty
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from PIL import ImageTk
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from src.core.data_types import Action, TerminationType
from src.core.logger import get_logger, cprint, MessageType
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.src.utils import process_image, process_compressed_image, get_output_path, process_odometry, process_twist
from src.core.utils import camelcase_to_snake_format, get_filename_without_extension

bridge = CvBridge()

COLOR_FG = "#141E61"
COLOR_BG = "#EEEEEE"
COLOR_VEL_0 = "#787A91"
COLOR_VEL_1 = "#0F044C"
FONT = "-*-lucidatypewriter-medium-r-*-*-*-140-*-*-*-*-*-*"

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
        self._rate_fps = 30
        self._border_width = 300
        self._counter = 0
        self._skip_first_n = 30
        self._skip_every_n = 4
        self._logger = get_logger(get_filename_without_extension(__file__), self._output_path)
        self._build_gui()
        self._subscribe()
        # Loop GUI
        self._window.mainloop()


    def _build_gui(self):
        self._window = Tk()
        self._window.title("Autonomous Navigation User Interface")
        self._window.geometry("500x400")
        # Frame for stats
        self._frame = Frame(self._window)
        self._frame.grid(row=0, column=0, sticky="n")
        self._fsm_label = Label(
            self._frame, text=f"", font=(FONT, 20), fg=COLOR_FG, bg=COLOR_BG
        )
        self._fsm_label.grid(column=0, row=0)
        self._battery_label = Label(
            self._frame, text=f"", font=(FONT, 20), fg=COLOR_FG, bg=COLOR_BG
        )
        self._battery_label.grid(column=0, row=1)

        self._cmd_label = Label(self._frame, text='', font=(FONT, 20), fg=COLOR_FG, bg=COLOR_BG)
        self._cmd_label.grid(row=2, column=0)

        self._wp_label = Label(self._frame, text='', font=(FONT, 20), fg=COLOR_FG, bg=COLOR_BG)
        self._wp_label.grid(row=3, column=0)

        # Frame for images
        self._img_canvas = Canvas(self._window, width=500, height=240)
        self._img_canvas.grid(row=1, column=0)
        self._img_canvas.place(relx=0.5, rely=0.5, anchor="center")


        self._img_canvas.configure(bg=COLOR_BG)
        dummy_img = PILImage.fromarray(np.random.randint(0, 255, size=(200,200,3), dtype=np.uint8))
        self._observation = ImageTk.PhotoImage(dummy_img)
        self._observation_label = Label(self._img_canvas, image=self._observation)
        self._observation_label.grid(row=0, column=0)
        
        dummy_mask = PILImage.fromarray(np.random.randint(0, 255, size=(200,200), dtype=np.uint8))
        self._mask = ImageTk.PhotoImage(dummy_mask)
        self._mask_label = Label(self._img_canvas, image=self._mask)
        self._mask_label.grid(row=0, column=1)
    
    def _subscribe(self):
        # Robot sensors:
        if rospy.has_param('/robot/camera_sensor'):
            sensor_topic = rospy.get_param('/robot/camera_sensor/topic')
            sensor_type = rospy.get_param('/robot/camera_sensor/type')
            sensor_callback = f'_process_image'
            if sensor_callback not in self.__dir__():
                cprint(f'Could not find sensor_callback {sensor_callback}', self._logger)
            sensor_stats = rospy.get_param('/robot/camera_sensor/stats', {})
            rospy.Subscriber(name=sensor_topic,
                             data_class=eval(sensor_type),
                             callback=self._process_image,
                             callback_args=("observation", sensor_stats))
            self._observation = None

        self._mask = None
        rospy.Subscriber(name='/mask', data_class=Image,
                             callback=self._process_image,
                             callback_args=("mask", {}))

        rospy.Subscriber('/fsm/reset', Empty, self._reset)

        # Applied action
        if rospy.has_param('/robot/command_topic'):
            rospy.Subscriber(name=rospy.get_param('/robot/command_topic'),
                             data_class=Twist,
                             callback=self._set_field,
                             callback_args=('action', {}))
        # fsm state
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

    def _reset(self, msg: Empty = None):
        self._reward = None
        self._fsm_state = None
        self._terminal_state = None

    def _process_image(self, msg: Union[Image, CompressedImage], args: tuple) -> None:
        field_name, sensor_stats = args
        image = process_image(msg, sensor_stats) if isinstance(msg, Image) else process_compressed_image(msg, sensor_stats)
        image = PILImage.fromarray((255. * image).astype(np.uint8).squeeze())
        if field_name == "observation":
            self._observation = ImageTk.PhotoImage(image)
            self._observation_label.configure(image=self._observation)
        elif field_name == "mask":
            self._mask = ImageTk.PhotoImage(image)
            self._mask_label.configure(image=self._mask)
        # cprint(f'set field {field_name}', self._logger, msg_type=MessageType.info)

    def _set_field(self, msg: Union[String, Twist, RosReward, CommonCommonStateBatteryStateChanged, Odometry],
                   args: Tuple) -> None:
        field_name, _ = args
        if field_name == 'fsm_state':
            self._fsm_label.configure(text=f"FSM state: {FsmState[msg.data].name}")
        elif field_name == 'action':
            cmd = process_twist(msg).value
            self._cmd_label.configure(text=f"Command: x: {cmd[0]:0.2f}, y: {cmd[1]:0.2f}, z: {cmd[2]:0.2f}, yaw: {cmd[5]:0.2f}")
        elif field_name == 'reward':
            self._reward = msg.reward
            self._terminal_state = TerminationType[msg.termination]
        elif field_name == 'reference_pose':
            self._reference_pose = np.asarray([msg.point.x, msg.point.y, msg.point.z])
            self._wp_label.configure(text=f"Reference point: x: {self._reference_pose[0]:0.2f}, y: {self._reference_pose[1]:0.2f}, z: {self._reference_pose[2]:0.2f}")
        elif field_name == 'battery':
            self._battery_label.configure(text=f'Battery level: {msg.percent}%')
        elif field_name == 'trajectory':
            global_pose = process_odometry(msg)
            self._trajectory.append(global_pose)
            # self._update_wp()
        else:
            raise NotImplementedError
        #cprint(f'set field {field_name}', self._logger, msg_type=MessageType.info)

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
