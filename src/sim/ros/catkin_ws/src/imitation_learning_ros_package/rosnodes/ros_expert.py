import os
import time
from collections import OrderedDict

import numpy as np
import rospy
import yaml
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image
from tf.transformations import euler_from_quaternion

from src.core.logger import get_logger, cprint
from src.sim.common.actors import Actor, ActorConfig
from src.sim.common.data_types import ActorType, Action
from src.sim.ros.src.utils import adapt_twist_to_action, process_laser_scan, process_image
from src.core.utils import camelcase_to_snake_format


class RosExpert(Actor):

    def __init__(self):
        while not rospy.has_param('/actor/ros_expert/specs'):
            time.sleep(0.01)
        specs = rospy.get_param('/actor/ros_expert/specs')
        super().__init__(
            config=ActorConfig(
                name='ros_expert',
                type=ActorType.Expert,
                specs=specs
            )
        )
        self._output_path = rospy.get_param('output_path', '/tmp')
        self._logger = get_logger('ros_expert', self._output_path)
        cprint(f'&&&&&&&&&&&&&&&&&& \n {self._specs} \n &&&&&&&&&&&&&&&&&', self._logger)
        with open(os.path.join(self._output_path, 'ros_expert_specs.yml'), 'w') as f:
            yaml.dump(self._specs, f)

        self._adjust_height = 0
        self._adjust_yaw_collision_avoidance = 0
        self._adjust_yaw_waypoint_following = 0
        self._reference_height = rospy.get_param('/world/starting_height', -1)
        self._waypoints = []
        self._current_waypoint_index = -1

        self._publisher = rospy.Publisher('/actor/ros_expert', Twist, queue_size=10)
        self._subscribe()
        rospy.init_node('ros_expert')

    def _subscribe(self):
        # Robot sensors:
        for sensor in ['depth_scan', 'pose_estimation']:
            if rospy.has_param(f'{sensor}_topic'):
                sensor_topic = rospy.get_param(f'{sensor}_topic')
                sensor_type = eval(rospy.get_param(f'{sensor}_type'))
                sensor_callback = f'self._process_{camelcase_to_snake_format(sensor_type)}'
                if not hasattr(self, sensor_callback):
                    cprint(f'Could not find sensor_callback {sensor_callback}', self._logger)
                sensor_stats = rospy.get_param(f'{sensor}_stats') if rospy.has_param(f'{sensor}_stats') else {}
                rospy.Subscriber(name=sensor_topic,
                                 data_class=sensor_type,
                                 callback=eval(sensor_callback),
                                 callback_args=(sensor_topic, sensor_stats))

    def _set_yaw_from_depth_map(self, depth_map: np.ndarray, field_of_view: float, front_width: float) -> None:
        map_depth_to_direction = OrderedDict()
        left_boundary = int(field_of_view / 2 - front_width / 2)
        right_boundary = int(field_of_view / 2 + front_width / 2)
        map_depth_to_direction['straight'] = np.nanmin(depth_map[left_boundary: right_boundary])
        map_depth_to_direction['left'] = np.nanmin(depth_map[:left_boundary])
        map_depth_to_direction['right'] = np.nanmin(depth_map[right_boundary:])
        compensations = {
            'straight': 0,
            'left': 1,
            'right': -1
        }
        self._adjust_yaw_collision_avoidance = compensations[max(map_depth_to_direction)]

    def _process_laser_scan(self, msg: LaserScan, args: tuple) -> None:
        sensor_topic, sensor_stats = args
        if 'max_depth' in self._specs.keys():
            sensor_stats['max_depth'] = self._specs['max_depth']
        if 'field_of_view' in self._specs.keys():
            sensor_stats['field_of_view'] = self._specs['field_of_view']
        sensor_stats['num_smooth_bins'] = 1  # use full resolution

        processed_scan = process_laser_scan(msg, sensor_stats)
        processed_scan[processed_scan == sensor_stats['max_depth']] = np.nan

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
        self._set_yaw_from_depth_map(depth_map=processed_depth_image,
                                     field_of_view=sensor_stats['field_of_view'],
                                     front_width=self._specs['front_width'])
        # TODO set adjust_height from depth map < vertical field-of-view and vertical frontwidth

    def _set_height(self, z: float):
        if self._reference_height == -1:
            return
        if z < (self._reference_height - 0.1):
            self._adjust_height = +1
        elif z > (self._reference_height + 0.1):
            self._adjust_height = -1
        else:
            self._adjust_height = 0

    def _process_odometry(self, msg: Odometry, args: tuple) -> None:
        sensor_topic, sensor_stats = args

        self._set_height(z=msg.pose.pose.position.z)

        if len(self._waypoints) == 0:
            return

        # adjust orientation towards current_waypoint
        quaternion = (msg.pose.pose.orientation.x,
                      msg.pose.pose.orientation.y,
                      msg.pose.pose.orientation.z,
                      msg.pose.pose.orientation.w)
        _, _, yaw_drone = euler_from_quaternion(quaternion)

        dy = (self._waypoints[self._current_waypoint_index][1] - msg.pose.pose.position.y)
        dx = (self._waypoints[self._current_waypoint_index][0] - msg.pose.pose.position.x)

        if np.sqrt(dx ** 2 + dy ** 2) < self._specs['waypoint_reached_distance']:
            # update to next waypoint:
            self._current_waypoint_index += 1
            self._current_waypoint_index = self._current_waypoint_index % len(self._waypoints)
            cprint(f"Reached waypoint: {self._current_waypoint_index-1}, "
                   f"next waypoint @ {self._waypoints[self._current_waypoint_index]}.", self._logger)
            self._adjust_yaw_waypoint_following = 0
            return
        else:
            # adjust for quadrants...
            yaw_goal = np.arctan(dy / dx)
            if np.sign(dx) == -1 and np.sign(dy) == +1:
                yaw_goal += np.pi
                # print("adjusted yaw_goal to 2th quadrant: {0} > 0".format(yaw_goal))
            elif np.sign(dx) == -1 and np.sign(dy) == -1:
                yaw_goal -= np.pi
                # print("adjusted yaw_goal to 3th quadrant: {0} < 0".format(yaw_goal))
            if np.abs(yaw_goal - yaw_drone) > self._specs['max_yaw_deviation_waypoint'] * np.pi / 180:
                adjust_yaw_goto = np.sign(yaw_goal - yaw_drone)
                # if difference between alpha and beta is bigger than pi:
                # swap direction because the other way is shorter.
                if np.abs(yaw_goal - yaw_drone) > np.pi:
                    self._adjust_yaw_waypoint_following = -1 * adjust_yaw_goto
            else:
                self._adjust_yaw_waypoint_following = 0

    def _update_twist(self):
        twist = Twist()
        twist.linear.x = self._specs['speed']
        twist.linear.z = self._adjust_height
        twist.angular.z = self._specs['collision_avoidance_weight'] * self._adjust_yaw_collision_avoidance \
            + self._specs['waypoint_following_weight'] * self._adjust_yaw_waypoint_following
        return twist

    def get_action(self, sensor_data: dict = None) -> Action:
        assert sensor_data is None
        action = adapt_twist_to_action(self._update_twist())
        action.actor_name = self._name
        action.actor_type = self._type
        return action

    def run(self):
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            self._publisher.publish(self._update_twist())
            rate.sleep()


if __name__ == "__main__":
    ros_expert = RosExpert()
    ros_expert.run()
