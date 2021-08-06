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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from PIL import ImageTk
from PIL import Image as PILImage
import matplotlib.pyplot as plt
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
        self._rate_fps = 10
        self._border_width = 300
        self._counter = 0
        self._skip_first_n = 30
        self._skip_every_n = 4
        self._logger = get_logger(get_filename_without_extension(__file__), self._output_path)
        self._subscribe()
        self._build_gui()

    def _build_gui(self):
        self._window = Tk()
        self._window.title("Autonomous Navigation User Interface")
        self._window.geometry("500x800")
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
        
        # Frame for images
        self._img_canvas = Canvas(self._window, width=480, height=240)
        self._img_canvas.grid(row=1, column=0)
        self._img_canvas.configure(bg=COLOR_BG)
        dummy_img = PILImage.fromarray(np.random.randint(0, 255, size=(200,200,3), dtype=np.uint8))
        self._observation = ImageTk.PhotoImage(dummy_img)
        self._observation_label = Label(self._img_canvas, image=self._observation)
        self._observation_label.grid(row=0, column=0)
        
        dummy_mask = PILImage.fromarray(np.random.randint(0, 255, size=(200,200), dtype=np.uint8))
        self._mask = ImageTk.PhotoImage(dummy_mask)
        self._mask_label = Label(self._img_canvas, image=self._mask)
        self._mask_label.grid(row=0, column=1)
    
        # Draw canvas for velocities
        self._cmd_canvas = Canvas(self._window, width=480, height=260)
        self._cmd_canvas.grid(row=2, column=0)
        self._cmd_canvas.configure(bg=COLOR_BG)

        # bar_width = 20
        # sec_bar_width = 15
        # bar_length = 200
        # max_vel = 1
        # x_vel = 0.5

        # x velocity
        # cmd_canvas.create_rectangle(
        #     120 - int(bar_width / 2),
        #     20,
        #     120 + int(bar_width / 2),
        #     20 + bar_length,
        #     fill=COLOR_VEL_0,
        # )
        # cmd_canvas.create_rectangle(
        #     120 - int(bar_width / 2),
        #     20,
        #     120 + int(bar_width / 2),
        #     20 + bar_length // 2,
        #     fill=COLOR_VEL_0,
        # )
        # self._x_bar = cmd_canvas.create_rectangle(
        #     120 - sec_bar_width // 2,
        #     20 + bar_length // 2,
        #     120 + sec_bar_width // 2,
        #     20 + (bar_length // 2) - x_vel * (bar_length // 2),
        #     fill=COLOR_VEL_1,
        # )
        # cmd_canvas.create_text(120, 10, text=f"x: {x_vel}", font=(FONT, 20))

        # y velocity
        # cmd_canvas.create_rectangle(
        #     20, 235 - bar_width // 2, 20 + bar_length, 235 + bar_width // 2, fill=COLOR_VEL_0,
        # )
        # cmd_canvas.create_rectangle(
        #     20,
        #     235 - bar_width // 2,
        #     20 + bar_length // 2,
        #     235 + bar_width // 2,
        #     fill=COLOR_VEL_0,
        # )
        # cmd_canvas.create_rectangle(
        #     20 + bar_length // 2,
        #     235 - sec_bar_width // 2,
        #     20 + (bar_length // 2) + y_vel * (bar_length // 2),
        #     235 + sec_bar_width // 2,
        #     fill=COLOR_VEL_1,
        # )
        # cmd_canvas.create_text(
        #     10, 205, text=f"y: {y_vel}", anchor="nw", font=(FONT, 20), fill=COLOR_FG
        # )

        # # z velocity
        # n_z_vel = z_vel / max_vel
        # cmd_canvas.create_rectangle(
        #     350 - bar_width // 2, 20, 350 + bar_width // 2, 20 + bar_length, fill=COLOR_VEL_0,
        # )
        # cmd_canvas.create_rectangle(
        #     350 - bar_width // 2,
        #     20,
        #     350 + bar_width // 2,
        #     20 + bar_length // 2,
        #     fill=COLOR_VEL_0,
        # )
        # cmd_canvas.create_rectangle(
        #     350 - sec_bar_width // 2,
        #     20 + bar_length // 2,
        #     350 + sec_bar_width // 2,
        #     20 + (bar_length // 2) - int(n_z_vel * (bar_length // 2)),
        #     fill=COLOR_VEL_1,
        # )
        # cmd_canvas.create_text(350, 10, text=f"z: {z_vel}", font=(FONT, 20), fill=COLOR_FG)

        # # yaw velocity
        # cmd_canvas.create_rectangle(
        #     250,
        #     235 - int(bar_width / 2),
        #     250 + bar_length,
        #     235 + int(bar_width / 2),
        #     fill=COLOR_VEL_0,
        # )
        # cmd_canvas.create_rectangle(
        #     250,
        #     235 - int(bar_width / 2),
        #     250 + bar_length // 2,
        #     235 + int(bar_width / 2),
        #     fill=COLOR_VEL_0,
        # )
        # cmd_canvas.create_rectangle(
        #     250 + bar_length // 2,
        #     235 - sec_bar_width // 2,
        #     250 + (bar_length // 2) + yaw_vel * (bar_length // 2),
        #     235 + sec_bar_width // 2,
        #     fill=COLOR_VEL_1,
        # )
        # cmd_canvas.create_text(
        #     240, 205, text="\u03c8" + str(yaw_vel), anchor="nw", font=(FONT, 20), fill=COLOR_FG
        # )

        # Draw waypoints
        self._wp_canvas = Canvas(self._window, width=480, height=260)
        self._wp_canvas.grid(row=3, column=0)
        self._wp_canvas.configure(bg=COLOR_BG)

        self._wp_fig, self._wp_ax = plt.subplots(1, 2, figsize=(9, 4), dpi=50)
        self._wp_ax[0].set_xlim(-4, 4)
        self._wp_ax[0].set_ylim(0, 4)
        self._wp_ax[0].set_title("Reference Point")
        self._reference_line, = self._wp_ax[0].plot([], [], 'o-', lw=2)
        self._wp_ax[1].set_title("Trajectory")

        self._chart_type = FigureCanvasTkAgg(self._wp_fig, self._wp_canvas)
        self._chart_type.get_tk_widget().pack()
        self._window.mainloop()

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

    def _update(self):
        print("updating window")
        # draw_images(self._window, image=self._observation, mask=self._mask)

        # draw_waypoints(self._window, self._reference_pose, self._trajectory)
        # self._window.update()
        self._window.after(100, self._update)        

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
        cprint(f'set field {field_name}', self._logger, msg_type=MessageType.info)

    def _set_field(self, msg: Union[String, Twist, RosReward, CommonCommonStateBatteryStateChanged, Odometry],
                   args: Tuple) -> None:
        field_name, _ = args
        if field_name == 'fsm_state':
            self._fsm_label.configure(text=f"FSM state: {FsmState[msg.data].name}")
        elif field_name == 'action':
            cmd = process_twist(msg).value
            self._draw_velocity_commands(cmd[0], cmd[1], cmd[2], cmd[5])
        elif field_name == 'reward':
            self._reward = msg.reward
            self._terminal_state = TerminationType[msg.termination]
        elif field_name == 'reference_pose':
            self._reference_pose = np.asarray([msg.point.x, msg.point.y, msg.point.z])
            self._update_wp()
        elif field_name == 'battery':
            self._battery_label.configure(text=f'Battery level: {msg.percent}%')
        elif field_name == 'trajectory':
            global_pose = process_odometry(msg)
            self._trajectory.append(global_pose)
            self._update_wp()
        else:
            raise NotImplementedError
        cprint(f'set field {field_name}', self._logger, msg_type=MessageType.info)


    def _update_wp(self):
        # Reference point
        if self._reference_pose is not None:
            self._reference_line.set_data(([0, self._reference_pose[1]], [0, self._reference_pose[0]]))
            # self._wp_ax[0].clear()
            # self._wp_ax[0].annotate(
            #     "",
            #     xy=(self._reference_pose[1], self._reference_pose[0]),
            #     xytext=(0, 0),
            #     arrowprops=dict(arrowstyle="->"),
            # )

        # Trajectory
        if self._trajectory is not None:
            self._wp_ax[1].scatter([p[0] for p in self._trajectory], [p[1] for p in self._trajectory])


    def _draw_velocity_commands(self, x_vel: float = 0., y_vel: float = 0., z_vel: float = 0., yaw_vel: float = 0.):
        pass        


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
