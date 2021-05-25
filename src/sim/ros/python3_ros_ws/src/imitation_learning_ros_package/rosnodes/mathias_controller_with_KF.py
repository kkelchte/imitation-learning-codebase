#!/usr/bin/python3.8
import operator
import os
import time
from collections import OrderedDict
from copy import copy

import matplotlib
matplotlib.use("agg")
import rospy
import yaml
from cv2 import cv2
from std_msgs.msg import Float32MultiArray, String, Header
from geometry_msgs.msg import Twist, PointStamped, Point, Pose, TwistStamped, Quaternion, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image
import actionlib
from hector_uav_msgs.msg import *
from scipy.spatial.transform import Rotation as R
from dynamic_reconfigure.server import Server
import matplotlib.pyplot as plt

from src.core.logger import get_logger, cprint, MessageType
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.actors import Actor, ActorConfig
from src.sim.common.noise import *
from src.core.data_types import Action, SensorType
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from imitation_learning_ros_package.cfg import pidConfig

from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.mathias_bebop_model import BebopModel
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.mathias_kalman_filter import KalmanFilter
from src.sim.ros.src.utils import process_laser_scan, process_image, euler_from_quaternion, \
    get_output_path, apply_noise_to_twist, process_twist, transform, get_timestamp, calculate_relative_orientation
from src.core.utils import camelcase_to_snake_format, get_filename_without_extension


class MathiasController:

    def __init__(self):
        self.count = 0
        rospy.init_node('controller')
        stime = time.time()
        max_duration = 60
        while not rospy.has_param('/actor/mathias_controller/specs') and time.time() < stime + max_duration:
            time.sleep(0.01)
        self._specs = rospy.get_param('/actor/mathias_controller/specs')

        self._output_path = get_output_path()
        self._logger = get_logger(get_filename_without_extension(__file__), self._output_path)
        cprint(f'controller specifications: {self._specs}', self._logger)
        with open(os.path.join(self._output_path, 'controller_specs.yml'), 'w') as f:
            yaml.dump(self._specs, f)
        self._reference_height = rospy.get_param('/world/starting_height', 1)

        self._rate_fps = self._specs['rate_fps'] if 'rate_fps' in self._specs.keys() else 15
        self.max_input = self._specs['max_input'] if 'max_input' in self._specs.keys() else 1.0
        # Not sure if this sample time should correspond to update rate of controller or measurement rate
        self._control_period = 1. / self._rate_fps
        self.Kp_x = self._specs['Kp_x'] if 'Kp_x' in self._specs.keys() else .5
        self.Ki_x = self._specs['Ki_x'] if 'Ki_x' in self._specs.keys() else 0.
        self.Kd_x = self._specs['Kd_x'] if 'Kd_x' in self._specs.keys() else .5
        self.Kp_y = self._specs['Kp_y'] if 'Kp_y' in self._specs.keys() else .5
        self.Ki_y = self._specs['Ki_y'] if 'Ki_y' in self._specs.keys() else 0.
        self.Kd_y = self._specs['Kd_y'] if 'Kd_y' in self._specs.keys() else .5
        self.Kp_z = self._specs['Kp_z'] if 'Kp_z' in self._specs.keys() else 1.
        self.Ki_z = self._specs['Ki_z'] if 'Ki_z' in self._specs.keys() else .5
        self.Kd_z = self._specs['Kd_z'] if 'Kd_z' in self._specs.keys() else 0.
        self.K_theta = self._specs['K_theta'] if 'K_theta' in self._specs.keys() else 1.0
        self._robot = rospy.get_param('/robot/model_name')
        self.model = BebopModel()
        self.filter = KalmanFilter(model=self.model)

        # if 'real' in self._robot:
        # Velocity in yaw is not measured on real bebop drone
        # Therefore, measurements should be ignored (weight of innovation = 1/mease_noise_cov)
        self.filter.meas_noise_cov[7, 7] = 10e8

        self.show_graph = self._specs['show_graph'] if 'show_graph' in self._specs.keys() else True
        self.data = {title: {
            axis: {label: [] for label in ['predicted', 'observed', 'adjusted']} for axis in ['x', 'y', 'z', 'yaw']}
                     for title in ['pose', 'velocity', 'command']
        }

        self.visualiser = rospy.Publisher('visualisation', Image, queue_size=10)
        self.last_measurement = None  # used to invoke slower measurement update in simulation

        self.desired_yaw = None
        self.real_yaw = None
        noise_config = self._specs['noise'] if 'noise' in self._specs.keys() else {}
        self._noise = eval(f"{noise_config['name']}(**noise_config['args'])") if noise_config else None
        self.last_cmd = TwistStamped(header=Header(stamp=rospy.Time().now()))
        self.pose_ref = None
        # expressed in rotated global frame according to drone's yaw
        self.vel_ref = TwistStamped(header=Header(frame_id="global_rotated"))
        self.prev_pose_error = PointStamped(header=Header(frame_id="global"))
        self.prev_vel_error = PointStamped(header=Header(frame_id="global_rotated"))

        self.pose_est = PoseStamped(header=Header(frame_id="global"),
                                    pose=Pose(orientation=Quaternion(w=1)))
        self.vel_est = PointStamped(header=Header(frame_id="global_rotated"))

        self._publisher = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self._subscribe()

    def _subscribe(self):
        self._fsm_state = FsmState.Unknown
        rospy.Subscriber(name='/fsm/state', data_class=String, callback=self._set_fsm_state)

        # listen to desired next reference point
        rospy.Subscriber('/reference_pose', PointStamped, self._reference_update)
        rospy.Subscriber(name='/waypoint_indicator/current_waypoint',
                         data_class=Float32MultiArray,
                         callback=self._waypoint_callback)

        # Robot sensors:
        for sensor in [SensorType.position]:
            if rospy.has_param(f'/robot/{sensor.name}_sensor/topic'):
                sensor_topic = rospy.get_param(f'/robot/{sensor.name}_sensor/topic')
                sensor_type = rospy.get_param(f'/robot/{sensor.name}_sensor/type')
                sensor_callback = f'_process_{camelcase_to_snake_format(sensor_type)}'
                if sensor_callback not in self.__dir__():
                    cprint(f'Could not find sensor_callback {sensor_callback}', self._logger)
                sensor_stats = rospy.get_param(f'/robot/{sensor.name}_sensor/stats') \
                    if rospy.has_param(f'/robot/{sensor.name}_sensor/stats') else {}
                rospy.Subscriber(name=sensor_topic,
                                 data_class=eval(sensor_type),
                                 callback=eval(f'self.{sensor_callback}'),
                                 callback_args=(sensor_topic, sensor_stats))

        # Dynamic reconfig server --> use only during tweaking
        self._config_server = Server(pidConfig, self._dynamic_config_callback)

    def _reset(self):
        self.data = {title: {
            axis: {label: [] for label in ['predicted', 'observed', 'adjusted']} for axis in ['x', 'y', 'z', 'yaw']}
            for title in ['pose', 'velocity', 'command']
        }
        self.prev_pose_error = PointStamped(header=Header(frame_id="global"))
        self.prev_vel_error = PointStamped(header=Header(frame_id="global_rotated"))
        self.last_cmd = TwistStamped(header=Header(stamp=rospy.Time().now()))

        if self.pose_ref is not None:
            _, _, yaw = euler_from_quaternion(self.pose_est.pose.orientation)
            self.desired_yaw = yaw + calculate_relative_orientation(robot_pose=self.pose_est,
                                                                    reference_pose=self.pose_ref)

    def _set_fsm_state(self, msg: String):
        # detect transition
        if self._fsm_state != FsmState[msg.data]:
            self._fsm_state = FsmState[msg.data]
            if self._fsm_state == FsmState.Running:  # reset early feedback values
                self._reset()
            elif self._fsm_state == FsmState.TakenOver and self.show_graph:
                self._plot()

    def _plot(self):
        colors = {'x': 'C0', 'y': 'C1', 'z': 'C2', 'yaw': 'C3'}
        markers = {'predicted': '.', 'observed': '+', 'adjusted': '*'}
        size = {'predicted': 8, 'observed': 30, 'adjusted': 20}
        y_labels = {'pose': '[m] or [rad]',
                    'velocity': '[m/s] or [rad/s]',
                    'command': '[m/s] or [rad/s]'}
        fig, axes = plt.subplots(nrows=len(self.data.keys()),
                                 ncols=1,
                                 sharex="all",
                                 squeeze=True,
                                 figsize=(25, 10 * len(self.data.keys())))
        for index, title in enumerate(self.data.keys()):
            axes[index].set_title(f'{title} in rotated global frame following yaw.')
            for direction in self.data[title].keys():
                for dt in self.data[title][direction].keys():
                    axes[index].plot([_[0] for _ in self.data[title][direction][dt]],
                                     [_[1] for _ in self.data[title][direction][dt]],
                                     marker=markers[dt], color=colors[direction],
                                     label=f'{direction}: {dt}',
                                     markersize=size[dt])
            axes[index].legend()
            axes[index].set_xlabel('time [s]', fontsize=10)
            axes[index].set_ylabel(y_labels[title], fontsize=10)

        fig.canvas.draw()
        fig.tight_layout(pad=0)
        data = np.fromstring(fig.canvas.tostring_rgb(),
                             dtype=np.uint8, sep='')
        frame = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        image = Image()
        image.data = frame.astype(np.uint8).flatten().tolist()
        image.height = frame.shape[0]
        image.width = frame.shape[1]
        image.encoding = 'rgb8'
        self.visualiser.publish(image)

    def _dynamic_config_callback(self, config, level):
        cprint(f'received config: {config}, level: {level}', self._logger)
        self._rate_fps = config['rate_fps']
        self._control_period = 1 / self._rate_fps
        self.max_input = config['max_input']
        self.Kp_x = config['Kp_x']
        self.Ki_x = config['Ki_x']
        self.Kd_x = config['Kd_x']
        self.Kp_y = config['Kp_y']
        self.Ki_y = config['Ki_y']
        self.Kd_y = config['Kd_y']
        self.Kp_z = config['Kp_z']
        self.Ki_z = config['Ki_z']
        self.Kd_z = config['Kd_z']
        self.K_theta = config['K_theta']
        return config

    def _store_datapoint(self, data: Union[Odometry, TwistStamped], label: str = "observed"):
        if not self.show_graph:
            return
        if isinstance(data, Odometry):
            # Note that pose is expressed in global frame ==> rotate to rotated global frame
            # While velocity is expressed in rotated global frame following yaw.
            stamp = float(data.header.stamp.to_sec())
            _, _, yaw = euler_from_quaternion((data.pose.pose.orientation.x,
                                               data.pose.pose.orientation.y,
                                               data.pose.pose.orientation.z,
                                               data.pose.pose.orientation.w))
            rotated_position = transform(points=[data.pose.pose.position],
                                         orientation=self.pose_est.pose.orientation,
                                         invert=True)[0]
            self.data['pose']['x'][label].append((stamp, rotated_position.x))
            self.data['pose']['y'][label].append((stamp, rotated_position.y))
            self.data['pose']['z'][label].append((stamp, rotated_position.z))
            self.data['pose']['yaw'][label].append((stamp, yaw))
            self.data['velocity']['x'][label].append((stamp, data.twist.twist.linear.x))
            self.data['velocity']['y'][label].append((stamp, data.twist.twist.linear.y))
            self.data['velocity']['z'][label].append((stamp, data.twist.twist.linear.z))
            self.data['velocity']['yaw'][label].append((stamp, data.twist.twist.angular.z))
        elif isinstance(data, TwistStamped):
            stamp = float(data.header.stamp.to_sec())
            if len(self.data['command']['x']['observed']) > 1 and \
                    abs(stamp - self.data['command']['x']['observed'][-1][0]) < 0.1:
                return
            self.data['command']['x'][label].append((stamp, data.twist.linear.x))
            self.data['command']['y'][label].append((stamp, data.twist.linear.y))
            self.data['command']['z'][label].append((stamp, data.twist.linear.z))
            self.data['command']['yaw'][label].append((stamp, data.twist.angular.z))

    def _process_odometry(self, msg: Odometry, args: tuple) -> None:
        sensor_topic, sensor_stats = args
        # impose 7Hz to avoid 1000Hz simulation updates
        if self.last_measurement is not None:
            difference = get_timestamp(msg) - self.last_measurement

            if difference < 1./7:
                return
        self.last_measurement = get_timestamp(msg)

        # simulated robots use /gt_states where twist message of odometry is expressed globally instead of locally
        if 'real' not in self._robot:
            _, _, yaw = euler_from_quaternion(msg.pose.pose.orientation)
            msg.twist.twist.linear = transform(points=[msg.twist.twist.linear],
                                               orientation=R.from_euler('XYZ', (0, 0, yaw)).as_matrix(),
                                               invert=True)[0]
        result = self.filter.kalman_correction(msg, self._control_period)
        if result is not None:
            self._store_datapoint(msg, 'observed')
            self._store_datapoint(result, 'adjusted')
            self.pose_est.pose = result.pose.pose
            self.vel_est.point.x = result.twist.twist.linear.x
            self.vel_est.point.y = result.twist.twist.linear.y
            self.vel_est.point.z = result.twist.twist.linear.z

    def _waypoint_callback(self, message: Float32MultiArray):
        pose_ref = message.data
        pose_ref = PointStamped(header=Header(frame_id="global"),
                                point=Point(x=pose_ref[0],
                                            y=pose_ref[1],
                                            z=self.pose_est.pose.position.z if len(pose_ref) == 2 else pose_ref[2]))
        self._reference_update(pose_ref)

    def _reference_update(self, pose_ref: PointStamped):
        if pose_ref.header.frame_id == "agent":
            cprint(f'got pose_ref in agent frame: {pose_ref.point}', self._logger)
            # TODO:
            # Use time stamp of message to project reference point to global frame
            # corresponding to the agent's pose at that time
            # Although, as references should be said once per second the small delay probably won't matter that much.

            # transform from agent to world frame.
            # Note that agent frame is not necessarily the same as agent_horizontal, so a small error is induced here
            # as pose_estimate.orientation only incorporates yaw turn because KF does not maintain roll and pitch.
            pose_ref.point = transform(points=[pose_ref.point],
                                       orientation=self.pose_est.pose.orientation,
                                       translation=self.pose_est.pose.position)[0]
            pose_ref.header.frame_id = "global"
            cprint(f'mapped to global frame: {pose_ref.point}', self._logger)
        if pose_ref != self.pose_ref:
            # reset pose error when new reference comes in.
            self.prev_pose_error = PointStamped(header=Header(frame_id="global"))
            self.prev_vel_error = PointStamped(header=Header(frame_id="global_rotated"))
            self.last_cmd = TwistStamped(header=Header(stamp=rospy.Time().now()))
            self.pose_ref = pose_ref
            _, _, yaw = euler_from_quaternion(self.pose_est.pose.orientation)
            self.desired_yaw = yaw + calculate_relative_orientation(robot_pose=self.pose_est,
                                                                    reference_pose=self.pose_ref)
            cprint(f'set pose_ref: {self.pose_ref.point}', self._logger)

    def _update_twist(self) -> Union[None, Twist]:
        result = self.filter.kalman_prediction(self.last_cmd, self._control_period)
        if result is None:
            return None
        self._store_datapoint(result, 'predicted')
        self.pose_est.pose = result.pose.pose
        self.vel_est.point.x = result.twist.twist.linear.x
        self.vel_est.point.y = result.twist.twist.linear.y
        self.vel_est.point.z = result.twist.twist.linear.z

        twist = self._feedback()
        if self._noise is not None:
            twist = apply_noise_to_twist(twist=twist, noise=self._noise.sample())
        return twist

    def run(self):
        while not rospy.is_shutdown():
            if self._fsm_state == FsmState.Running:
                twist = self._update_twist()
                if twist is not None:
                    self.last_cmd = TwistStamped(
                        header=Header(stamp=rospy.Time().now()),
                        twist=twist
                    )
                    self._store_datapoint(self.last_cmd)
                    self._publisher.publish(twist)
                self.count += 1
                if self.count % 10 * self._rate_fps == 0 and self.pose_est is not None and self.pose_ref is not None:
                    _, _, yaw = euler_from_quaternion(self.pose_est.pose.orientation)
                    msg = f'<<reference: {self.pose_ref.point}, \n<<pose: {self.pose_est.pose.position} \n ' \
                          f'<<yaw {yaw},  \ncontrol: {self.last_cmd.twist.linear}'
                    cprint(msg, self._logger)
            rospy.sleep(duration=1/self._rate_fps)

    def _feedback(self):
        '''Whenever the target is reached, apply position feedback to the
        desired end position to remain in the correct spot and compensate for
        drift. Tustin discretized PID controller for x and y, PI for z.
        Returns a twist containing the control command.
        '''
        cmd = Twist()
        if self.pose_ref is None:
            return cmd
        # PID
        prev_pose_error = self.prev_pose_error

        pose_error = PointStamped()
        pose_error.point.x = (self.pose_ref.point.x - self.pose_est.pose.position.x)
        pose_error.point.y = (self.pose_ref.point.y - self.pose_est.pose.position.y)
        pose_error.point.z = (self.pose_ref.point.z - self.pose_est.pose.position.z)
        pose_error.point = transform(points=[pose_error.point],
                                     orientation=self.pose_est.pose.orientation,
                                     invert=True)[0]
        pose_error.header.frame_id = "global_rotated"

        prev_vel_error = self.prev_vel_error
        vel_error = PointStamped()
        vel_error.header.frame_id = "global_rotated"
        vel_error.point.x = self.vel_ref.twist.linear.x - self.vel_est.point.x
        vel_error.point.y = self.vel_ref.twist.linear.y - self.vel_est.point.y

        # calculations happen in global_rotated frame
        cmd.linear.x = max(-self.max_input, min(self.max_input, (
                self.last_cmd.twist.linear.x +
                (self.Kp_x + self.Ki_x * self._control_period / 2) * pose_error.point.x +
                (-self.Kp_x + self.Ki_x * self._control_period / 2) * prev_pose_error.point.x +
                self.Kd_x * (vel_error.point.x - prev_vel_error.point.x))))

        cmd.linear.y = max(-self.max_input, min(self.max_input, (
                self.last_cmd.twist.linear.y +
                (self.Kp_y + self.Ki_y * self._control_period / 2) * pose_error.point.y +
                (-self.Kp_y + self.Ki_y * self._control_period / 2) * prev_pose_error.point.y +
                self.Kd_y * (vel_error.point.y - prev_vel_error.point.y))))

        cmd.linear.z = max(-self.max_input, min(self.max_input, (
                self.last_cmd.twist.linear.z +
                (self.Kp_z + self.Ki_z * self._control_period / 2) * pose_error.point.z +
                (-self.Kp_z + self.Ki_z * self._control_period / 2) * prev_pose_error.point.z +
                self.Kd_z * (vel_error.point.z - prev_vel_error.point.z))))  # Added derivative term

        # if target is more than 1m away, look in that direction
        _, _, yaw = euler_from_quaternion(self.pose_est.pose.orientation)
        if self.desired_yaw is not None:  # in case there is a desired yaw
            angle_error = ((((self.desired_yaw - yaw) - np.pi) % (2*np.pi)) - np.pi)
            K_theta = self.K_theta + (np.pi - abs(angle_error))/np.pi*0.2
            cmd.angular.z = (K_theta * angle_error)

        self.prev_pose_error = pose_error
        self.prev_vel_error = vel_error
        return cmd


if __name__ == "__main__":
    controller = MathiasController()
    controller.run()
