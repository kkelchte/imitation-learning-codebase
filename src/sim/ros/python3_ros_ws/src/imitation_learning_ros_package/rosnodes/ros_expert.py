#!/usr/bin/python3.8
import operator
import os
import time
from collections import OrderedDict

import rospy
import yaml
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image

from src.core.logger import get_logger, cprint, MessageType
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.actors import Actor, ActorConfig
from src.sim.common.noise import *
from src.core.data_types import Action, SensorType
from src.sim.ros.src.utils import process_laser_scan, process_image, euler_from_quaternion, \
    get_output_path, apply_noise_to_twist, process_twist
from src.core.utils import camelcase_to_snake_format, get_filename_without_extension


class RosExpert(Actor):

    def __init__(self):
        self.count = 0
        rospy.init_node('ros_expert')
        stime = time.time()
        max_duration = 60
        while not rospy.has_param('/actor/ros_expert/specs') and time.time() < stime + max_duration:
            time.sleep(0.01)
        self._specs = rospy.get_param('/actor/ros_expert/specs')
        super().__init__(
            config=ActorConfig(
                name='ros_expert',
                specs=self._specs
            )
        )
        self._output_path = get_output_path()
        self._logger = get_logger(get_filename_without_extension(__file__), self._output_path)
        cprint(f'ros specifications: {self._specs}', self._logger)
        with open(os.path.join(self._output_path, 'ros_expert_specs.yml'), 'w') as f:
            yaml.dump(self._specs, f)
        self._reference_height = rospy.get_param('/world/starting_height', 1)
        self._adjust_height = 0
        self._adjust_yaw_collision_avoidance = 0
        self._adjust_yaw_waypoint_following = 0
        self._rate_fps = self._specs['rate_fps'] if 'rate_fps' in self._specs.keys() else 10
        self._next_waypoint = []
        noise_config = self._specs['noise'] if 'noise' in self._specs.keys() else {}
        self._noise = eval(f"{noise_config['name']}(**noise_config['args'])") if noise_config else None

        self._publisher = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self._subscribe()

    def _subscribe(self):
        # Robot sensors:
        for sensor in [SensorType.depth,
                       SensorType.position]:
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

    def _set_yaw_from_depth_map(self, depth_map: np.ndarray, field_of_view: float, front_width: float) -> None:
        left_boundary = int(field_of_view / 2 - front_width / 2)
        right_boundary = int(field_of_view / 2 + front_width / 2)
        map_depth_to_direction = OrderedDict()
        map_depth_to_direction['straight'] = np.nanmin(depth_map[left_boundary: right_boundary])
        map_depth_to_direction['left'] = np.nanmin(depth_map[:left_boundary])
        map_depth_to_direction['right'] = np.nanmin(depth_map[right_boundary:])
        direction_with_max_depth = max(map_depth_to_direction.items(), key=operator.itemgetter(1))[0]
        compensating_yaw_values = {
            'straight': 0,
            'left': 1,
            'right': -1
        }
        self._adjust_yaw_collision_avoidance = compensating_yaw_values[direction_with_max_depth]

    def _process_laser_scan(self, msg: LaserScan, args: tuple) -> None:
        sensor_topic, sensor_stats = args
        if 'max_depth' in self._specs.keys():
            sensor_stats['max_depth'] = self._specs['max_depth']
        if 'field_of_view' in self._specs.keys():
            sensor_stats['field_of_view'] = self._specs['field_of_view']
        sensor_stats['num_smooth_bins'] = 1  # use full resolution

        processed_scan = process_laser_scan(msg, sensor_stats)
        if len(processed_scan) != 0:
            self._set_yaw_from_depth_map(depth_map=processed_scan,
                                         field_of_view=sensor_stats['field_of_view'],
                                         front_width=self._specs['front_width'])

    def _process_image(self, msg: Image, args: tuple) -> None:
        sensor_topic, sensor_stats = args
        if 'max_depth' in self._specs.keys():
            sensor_stats['max_depth'] = self._specs['max_depth']
        if 'field_of_view' in self._specs.keys():
            sensor_stats['field_of_view'] = self._specs['field_of_view']
        sensor_stats['num_smooth_bins'] = 1  # use full resolution
        processed_depth_image = process_image(msg, sensor_stats)
        # TODO processed_depth_image has to be reduced to 1d array as yaw is calculated on first matrix array
        if len(processed_depth_image) != 0:
            self._set_yaw_from_depth_map(depth_map=processed_depth_image,
                                         field_of_view=sensor_stats['field_of_view'],
                                         front_width=self._specs['front_width'])
        # TODO set adjust_height from depth map < vertical field-of-view and vertical frontwidth

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

        if not self._next_waypoint:
            return

        # adjust orientation towards current_waypoint
        quaternion = (msg.pose.pose.orientation.x,
                      msg.pose.pose.orientation.y,
                      msg.pose.pose.orientation.z,
                      msg.pose.pose.orientation.w)
        _, _, yaw_drone = euler_from_quaternion(quaternion)

        dx = (self._next_waypoint[0] - msg.pose.pose.position.x)
        dy = (self._next_waypoint[1] - msg.pose.pose.position.y)

        # adjust for quadrants...
        yaw_goal = np.arctan(dy / (dx + 1e-6))
        # cprint(f'yaw_goal: {yaw_goal}', self._logger)
        if np.sign(dx) == -1 and np.sign(dy) == +1:
            yaw_goal += np.pi
            # cprint("adjusted yaw_goal to 2th quadrant: {0} > 0".format(yaw_goal), self._logger)
        elif np.sign(dx) == -1 and np.sign(dy) == -1:
            yaw_goal -= np.pi
            # cprint("adjusted yaw_goal to 3th quadrant: {0} < 0".format(yaw_goal), self._logger)
        if np.abs(yaw_goal - yaw_drone) > self._specs['max_yaw_deviation_waypoint'] * np.pi / 180:
            # cprint(f"threshold reached: {np.abs(yaw_goal - yaw_drone)} >"
            #        f" {self._specs['max_yaw_deviation_waypoint'] * np.pi / 180}", self._logger)
            self._adjust_yaw_waypoint_following = np.sign(yaw_goal - yaw_drone)
            # if difference between alpha and beta is bigger than pi:
            # swap direction because the other way is shorter.
            if np.abs(yaw_goal - yaw_drone) > np.pi:
                self._adjust_yaw_waypoint_following = -1 * self._adjust_yaw_waypoint_following
        else:
            # cprint(f"threshold NOT reached: {np.abs(yaw_goal - yaw_drone)} <"
            #        f" {self._specs['max_yaw_deviation_waypoint'] * np.pi / 180}", self._logger)
            self._adjust_yaw_waypoint_following = 0
        # cprint(f'set adjust_yaw to {self._adjust_yaw_waypoint_following}')

    def _update_twist(self):
        yaw_velocity = self._specs['collision_avoidance_weight'] * self._adjust_yaw_collision_avoidance \
            + self._specs['waypoint_following_weight'] * self._adjust_yaw_waypoint_following
        twist = Twist()
        twist.linear.x = self._specs['speed'] if np.abs(yaw_velocity) < 0.1 or 'turn_speed' not in self._specs.keys() \
            else self._specs['turn_speed']
        twist.angular.z = yaw_velocity
        twist.linear.z = self._adjust_height

        if self._noise is not None:
            twist = apply_noise_to_twist(twist=twist, noise=self._noise.sample())
        # cprint(f'twist: {twist.linear.x}, {twist.linear.y}, {twist.linear.z}, '
        #        f'{twist.angular.x}, {twist.angular.y}, {twist.angular.z}', self._logger, msg_type=MessageType.debug)
        return twist

    def get_action(self, sensor_data: dict = None) -> Action:
        assert sensor_data is None
        action = process_twist(self._update_twist())
        action.actor_name = self._name
        # cprint(f'action: {action}', self._logger, msg_type=MessageType.debug)
        return action

    def run(self):
        rate = rospy.Rate(self._rate_fps)
        while not rospy.is_shutdown():
            self._publisher.publish(self._update_twist())
            self.count += 1
            if self.count % 10 * self._rate_fps == 0:
                msg = f'waypoint yaw adjustment: {self._adjust_yaw_waypoint_following} \n'
                msg += f' collision yaw adjustment: {self._adjust_yaw_collision_avoidance} \n'
                msg += f' next waypoint: {self._next_waypoint} \n'
                msg += f' cmd: {self._update_twist()}'
                cprint(msg, self._logger, msg_type=MessageType.info)
            rate.sleep()


if __name__ == "__main__":
    ros_expert = RosExpert()
    ros_expert.run()
