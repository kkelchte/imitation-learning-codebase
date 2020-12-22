#!/usr/bin/python3.8
import operator
import os
import time
from collections import OrderedDict

import rospy
import yaml
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import Twist, PointStamped, Point, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image
import actionlib
from hector_uav_msgs.msg import *
import tf
import tf2_ros
import tf2_geometry_msgs as tf2_geom
from scipy.spatial.transform import Rotation as R

from src.core.logger import get_logger, cprint, MessageType
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.actors import Actor, ActorConfig
from src.sim.common.noise import *
from src.core.data_types import Action, SensorType
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.src.utils import process_laser_scan, process_image, euler_from_quaternion, \
    get_output_path, apply_noise_to_twist, process_twist, transform
from src.core.utils import camelcase_to_snake_format, get_filename_without_extension


class MathiasController:

    def __init__(self):
        self.count = 0
        rospy.init_node('controller')
        stime = time.time()
        max_duration = 60
        while not rospy.has_param('/actor/controller/specs') and time.time() < stime + max_duration:
            time.sleep(0.01)
        self._specs = rospy.get_param('/actor/controller/specs')
        self._output_path = get_output_path()
        self._logger = get_logger(get_filename_without_extension(__file__), self._output_path)
        cprint(f'controller specifications: {self._specs}', self._logger)
        with open(os.path.join(self._output_path, 'controller_specs.yml'), 'w') as f:
            yaml.dump(self._specs, f)
        self._reference_height = rospy.get_param('/world/starting_height', 1)

        self._rate_fps = self._specs['rate_fps'] if 'rate_fps' in self._specs.keys() else 10
        self.max_input = self._specs['max_input'] if 'max_input' in self._specs.keys() else 0.5
        self._sample_time = self._specs['_sample_time'] if '_sample_time' in self._specs.keys() else  0.01
        self.Kp_x = self._specs['Kp_x'] if 'Kp_x' in self._specs.keys() else 0.6864
        self.Ki_x = self._specs['Ki_x'] if 'Ki_x' in self._specs.keys() else 0.6864
        self.Kd_x = self._specs['Kd_x'] if 'Kd_x' in self._specs.keys() else 0.6864
        self.Kp_y = self._specs['Kp_y'] if 'Kp_y' in self._specs.keys() else 0.6864
        self.Ki_y = self._specs['Ki_y'] if 'Ki_y' in self._specs.keys() else 0.6864
        self.Kd_y = self._specs['Kd_y'] if 'Kd_y' in self._specs.keys() else 0.6864
        self.Kp_z = self._specs['Kp_z'] if 'Kp_z' in self._specs.keys() else 0.5
        self.Ki_z = self._specs['Ki_z'] if 'Ki_z' in self._specs.keys() else 1.5792
        self.K_theta = self._specs['K_theta'] if 'K_theta' in self._specs.keys() else 0.3

        self.real_yaw = 0.0
        self.desired_yaw = np.pi / 2.
        self._next_waypoint = []
        self._fsm_state = FsmState.Unknown
        noise_config = self._specs['noise'] if 'noise' in self._specs.keys() else {}
        self._noise = eval(f"{noise_config['name']}(**noise_config['args'])") if noise_config else None
        self._publisher = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self._subscribe()

        self.pos_error_prev = 0
        self.vel_error_prev = 0
        self.drone_pose_est = Pose()
        self.drone_vel_est = Point()
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

    def _subscribe(self):
        rospy.Subscriber('/fsm/state', String, self._fsm_state_update)
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
        # Listen to next waypoint
        rospy.Subscriber(name='/waypoint_indicator/current_waypoint',
                         data_class=Float32MultiArray,
                         callback=self._update_waypoint)
        # Listen to fsm state
        rospy.Subscriber(name='/fsm/state',
                         data_class=Float32MultiArray,
                         callback=self._update_waypoint)

    def _fsm_state_update(self, msg: String):
        self.count = 0
        if self._fsm_state != FsmState[msg.data]:
            cprint(f'update fsm state to {FsmState[msg.data]}', self._logger, msg_type=MessageType.debug)
        self._fsm_state = FsmState[msg.data]

    def _set_height(self, z: float):
        if z < (self._reference_height - 0.1):
            self._adjust_height = self._specs['height_speed'] if 'height_speed' in self._specs.keys() else +0.5
        elif z > (self._reference_height + 0.1):
            self._adjust_height = -self._specs['height_speed'] if 'height_speed' in self._specs.keys() else -0.5
        else:
            self._adjust_height = 0

    def _update_waypoint(self, msg: Float32MultiArray):
        self._next_waypoint = msg.data
        if len(self._next_waypoint) > 2:
            self._reference_height = self._next_waypoint[2]

    def _process_odometry(self, msg: Odometry, args: tuple) -> None:
        sensor_topic, sensor_stats = args
        self._set_height(z=msg.pose.pose.position.z)

        # adjust orientation towards current_waypoint
        quaternion = (msg.pose.pose.orientation.x,
                      msg.pose.pose.orientation.y,
                      msg.pose.pose.orientation.z,
                      msg.pose.pose.orientation.w)
        _, _, self.real_yaw = euler_from_quaternion(quaternion)

        self.drone_pose_est = msg.pose.pose
        self.drone_vel_est.x = msg.twist.twist.linear.x
        self.drone_vel_est.y = msg.twist.twist.linear.y
        self.drone_vel_est.z = msg.twist.twist.linear.z

    def _follow_trajectory(self) -> Twist:
        return Twist()

    def _hover(self) -> Twist:
        pos_desired = PointStamped()
        pos_desired.point = Point(x=0,
                                  y=0,
                                  z=self._reference_height)

        return self._feedback(pos_desired, Twist())

    def _update_twist(self) -> Twist:
        twist = Twist()
        if self._fsm_state == FsmState.Running:
            twist = self._follow_trajectory()
            if self._noise is not None:
                twist = apply_noise_to_twist(twist=twist, noise=self._noise.sample())
        elif self._fsm_state == FsmState.Terminated:
            twist = self._hover()
        return twist

    def run(self):
        rate = rospy.Rate(self._rate_fps)
        while not rospy.is_shutdown():
            self._publisher.publish(self._update_twist())
            self.count += 1
            if self.count % 10 * self._rate_fps == 0:
                msg = f'control:  cmd: {self._update_twist()} \n'
                if len(self._next_waypoint) != 0:
                    msg += f' next waypoint: {self._next_waypoint} \n'
                cprint(msg, self._logger, msg_type=MessageType.info)
            rate.sleep()

    def _feedback(self, pos_desired, vel_desired):
        '''Whenever the target is reached, apply position feedback to the
        desired end position to remain in the correct spot and compensate for
        drift.
        Tustin discretized PID controller for x and y, PI for z.
        '''
        fb_cmd = Twist()

        # PID
        pos_error_prev = self.pos_error_prev
        pos_error = PointStamped()
        pos_error.header.frame_id = "world"
        pos_error.point.x = (pos_desired.point.x - self.drone_pose_est.position.x)
        pos_error.point.y = (pos_desired.point.y - self.drone_pose_est.position.y)
        pos_error.point.z = (pos_desired.point.z - self.drone_pose_est.position.z)

        vel_error_prev = self.vel_error_prev
        vel_error = PointStamped()
        vel_error.header.frame_id = "world"
        vel_error.point.x = vel_desired.linear.x - self.drone_vel_est.x
        vel_error.point.y = vel_desired.linear.y - self.drone_vel_est.y

        pos_error, vel_error = transform([np.asarray([pos_error.point.x, pos_error.point.y, pos_error.point.z]),
                                          np.asarray([vel_error.point.x, vel_error.point.y, vel_error.point.z])],
                                         R.from_euler('XYZ', (0, 0, self.real_yaw), degrees=False).as_matrix())

        fb_cmd.linear.x = max(-self.max_input, min(self.max_input, (
                              self.fb_cmd_prev.linear.x +
                              (self.Kp_x + self.Ki_x*self._sample_time/2) * pos_error.point.x +
                              (-self.Kp_x + self.Ki_x*self._sample_time/2) *
                              pos_error_prev.point.x +
                              self.Kd_x*(vel_error.point.x - vel_error_prev.point.x))))

        fb_cmd.linear.y = max(-self.max_input, min(self.max_input, (
                              self.fb_cmd_prev.linear.y +
                              (self.Kp_y + self.Ki_y*self._sample_time/2) *
                              pos_error.point.y +
                              (-self.Kp_y + self.Ki_y*self._sample_time/2) *
                              pos_error_prev.point.y +
                              self.Kd_y*(vel_error.point.y - vel_error_prev.point.y))))

        fb_cmd.linear.z = max(-self.max_input, min(self.max_input, (
                              self.fb_cmd_prev.linear.z +
                              (self.Kp_z + self.Ki_z*self._sample_time/2) *
                              pos_error.point.z +
                              (-self.Kp_z + self.Ki_z*self._sample_time/2) *
                              pos_error_prev.point.z)))

        # Add theta feedback to remain at zero yaw angle
        angle_error = ((((self.desired_yaw - self.real_yaw) - np.pi) % (2*np.pi)) - np.pi)
        K_theta = self.K_theta + (np.pi - abs(angle_error))/np.pi*0.2
        fb_cmd.angular.z = (K_theta*angle_error)

        self.pos_error_prev = pos_error
        self.vel_error_prev = vel_error
        self.fb_cmd_prev = fb_cmd
        return fb_cmd


if __name__ == "__main__":
    controller = MathiasController()
    controller.run()
