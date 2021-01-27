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
# import tf2_ros
# import tf2_geometry_msgs as tf2_geom
from scipy.spatial.transform import Rotation as R
from dynamic_reconfigure.server import Server

from src.core.logger import get_logger, cprint, MessageType
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.actors import Actor, ActorConfig
from src.sim.common.noise import *
from src.core.data_types import Action, SensorType
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from imitation_learning_ros_package.cfg import pidConfig
from src.sim.ros.src.utils import process_laser_scan, process_image, euler_from_quaternion, \
    get_output_path, apply_noise_to_twist, process_twist, transform
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

        self._rate_fps = self._specs['rate_fps'] if 'rate_fps' in self._specs.keys() else 60
        self.max_input = self._specs['max_input'] if 'max_input' in self._specs.keys() else 1.0
        self._sample_time = self._specs['_sample_time'] if '_sample_time' in self._specs.keys() else 1./self._rate_fps
        self.Kp_x = self._specs['Kp_x'] if 'Kp_x' in self._specs.keys() else 2.0
        self.Ki_x = self._specs['Ki_x'] if 'Ki_x' in self._specs.keys() else 0.2
        self.Kd_x = self._specs['Kd_x'] if 'Kd_x' in self._specs.keys() else 0.4
        self.Kp_y = self._specs['Kp_y'] if 'Kp_y' in self._specs.keys() else 2.0
        self.Ki_y = self._specs['Ki_y'] if 'Ki_y' in self._specs.keys() else 0.2
        self.Kd_y = self._specs['Kd_y'] if 'Kd_y' in self._specs.keys() else 0.4
        self.Kp_z = self._specs['Kp_z'] if 'Kp_z' in self._specs.keys() else 2.0
        self.Ki_z = self._specs['Ki_z'] if 'Ki_z' in self._specs.keys() else 0.2
        self.Kd_z = self._specs['Kd_z'] if 'Kd_z' in self._specs.keys() else 0.
        self.K_theta = self._specs['K_theta'] if 'K_theta' in self._specs.keys() else 0.3

        self.real_yaw = 0.0
        self.desired_yaw = 0.0  #np.pi / 8.
        self._fsm_state = FsmState.Unknown
        noise_config = self._specs['noise'] if 'noise' in self._specs.keys() else {}
        self._noise = eval(f"{noise_config['name']}(**noise_config['args'])") if noise_config else None
        self.prev_cmd = Twist()
        self.pose_est = Pose()
        self.vel_est = Point()
        self.pose_ref = PointStamped()
        self.vel_ref = Twist()
        self.prev_pose_error = PointStamped()
        self.prev_vel_error = PointStamped()

        self._publisher = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self._subscribe()

        # self.tfBuffer = tf2_ros.Buffer()
        # self.listener = tf2_ros.TransformListener(self.tfBuffer)

    def _subscribe(self):
        # listen to desired next reference point
        rospy.Subscriber('/reference_pose', PointStamped, self._reference_update)
        rospy.Subscriber(name='/waypoint_indicator/current_waypoint',
                         data_class=Float32MultiArray,
                         callback=self._reference_update)

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

        # Dynamic reconfig server
        self._config_server = Server(pidConfig, self._dynamic_config_callback)

    def _dynamic_config_callback(self, config, level):
        cprint(f'received config: {config}, level: {level}', self._logger)
        self._rate_fps = config['rate_fps']
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

    def _process_odometry(self, msg: Odometry, args: tuple) -> None:
        sensor_topic, sensor_stats = args
        # adjust orientation towards current_waypoint
        quaternion = (msg.pose.pose.orientation.x,
                      msg.pose.pose.orientation.y,
                      msg.pose.pose.orientation.z,
                      msg.pose.pose.orientation.w)
        _, _, self.real_yaw = euler_from_quaternion(quaternion)
        print(f'yaw world frame: {self.real_yaw}')

        self.pose_est = msg.pose.pose
        self.vel_est.x = msg.twist.twist.linear.x
        self.vel_est.y = msg.twist.twist.linear.y
        self.vel_est.z = msg.twist.twist.linear.z

    def _reference_update(self, message: PointStamped) -> None:
        if isinstance(message, PointStamped):
            pose_ref = message
        elif isinstance(message, Float32MultiArray):
            pose_ref = message.data
            pose_ref = PointStamped(point=Point(x=pose_ref[0],
                                                y=pose_ref[1],
                                                z=1 if len(pose_ref) == 2 else pose_ref[2]))
        self.pose_ref = pose_ref

    def _update_twist(self) -> Twist:
        twist = self._feedback()
        if self._noise is not None:
            twist = apply_noise_to_twist(twist=twist, noise=self._noise.sample())
        return twist

    def run(self):
        rate = rospy.Rate(self._rate_fps)
        while not rospy.is_shutdown():
            cmd = self._update_twist()
            self._publisher.publish(cmd)
            self.count += 1
            if self.count % 10 * self._rate_fps == 0:
                msg = f'<<reference: {self.pose_ref}, \n<<pose: {self.pose_est} \n control: {cmd}'
                cprint(msg, self._logger)
            rate.sleep()

    def _feedback(self):
        '''Whenever the target is reached, apply position feedback to the
        desired end position to remain in the correct spot and compensate for
        drift. Tustin discretized PID controller for x and y, PI for z.
        Returns a twist containing the control command.
        '''
        
        # PID
        prev_pose_error = self.prev_pose_error
        pose_error = PointStamped()
        pose_error.header.frame_id = "world"
        pose_error.point.x = (self.pose_ref.point.x - self.pose_est.position.x)
        pose_error.point.y = (self.pose_ref.point.y - self.pose_est.position.y)
        pose_error.point.z = (self.pose_ref.point.z - self.pose_est.position.z)

        prev_vel_error = self.prev_vel_error
        vel_error = PointStamped()
        vel_error.header.frame_id = "world"
        vel_error.point.x = self.vel_ref.linear.x - self.vel_est.x
        vel_error.point.y = self.vel_ref.linear.y - self.vel_est.y

        pose_error.point, vel_error.point = transform([pose_error.point, vel_error.point],
                                                      R.from_euler('XYZ', (0, 0, self.real_yaw),
                                                                   degrees=False).as_matrix())

        cmd = Twist()
        cmd.linear.x = max(-self.max_input, min(self.max_input, (
                              self.prev_cmd.linear.x +
                              (self.Kp_x + self.Ki_x * self._sample_time/2) * pose_error.point.x +
                              (-self.Kp_x + self.Ki_x * self._sample_time/2) * prev_pose_error.point.x +
                              self.Kd_x * (vel_error.point.x - prev_vel_error.point.x))))

        cmd.linear.y = max(-self.max_input, min(self.max_input, (
                              self.prev_cmd.linear.y +
                              (self.Kp_y + self.Ki_y * self._sample_time/2) * pose_error.point.y +
                              (-self.Kp_y + self.Ki_y * self._sample_time/2) * prev_pose_error.point.y +
                              self.Kd_y * (vel_error.point.y - prev_vel_error.point.y))))

        cmd.linear.z = max(-self.max_input, min(self.max_input, (
                              self.prev_cmd.linear.z +
                              (self.Kp_z + self.Ki_z * self._sample_time/2) * pose_error.point.z +
                              (-self.Kp_z + self.Ki_z * self._sample_time/2) * prev_pose_error.point.z +
                              self.Kd_z * (vel_error.point.z - prev_vel_error.point.z))))  # Added derivative term

        # Add theta feedback to remain at zero yaw angle
        # extension: look at reference point
        angle_error = ((((self.desired_yaw - self.real_yaw) - np.pi) % (2*np.pi)) - np.pi)
        K_theta = self.K_theta + (np.pi - abs(angle_error))/np.pi*0.2
        # angle_error = np.arctan(pose_error.point.y / pose_error.point.x) % 2 * np.pi
        cmd.angular.z = (K_theta * angle_error)

        self.prev_pose_error = pose_error
        self.prev_vel_error = vel_error
        self.prev_cmd = cmd
        return cmd


if __name__ == "__main__":
    controller = MathiasController()
    controller.run()
