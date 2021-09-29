#!/usr/bin/python3.8
from __future__ import print_function

import sys
import select
import termios
import time
import tty

import roslib
import rospy
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist
from hector_uav_msgs.srv import EnableMotors  # DO NOT REMOVE

from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.actors import Actor, ActorConfig
from src.sim.ros.src.utils import get_output_path

roslib.load_manifest('teleop_twist_keyboard')


class KeyboardActor(Actor):

    def __init__(self):
        rospy.init_node('teleop_twist_keyboard')
        self.camera_direction = 'straight'
        start_time = time.time()
        max_duration = 60
        while not rospy.has_param('/output_path') and time.time() < start_time + max_duration:
            time.sleep(0.1)
        self.specs = rospy.get_param('/actor/keyboard/specs')
        super().__init__(
            config=ActorConfig(
                name='keyboard',
                specs=self.specs
            )
        )
        self.settings = termios.tcgetattr(sys.stdin)
        self._logger = get_logger(get_filename_without_extension(__file__), get_output_path())

        self.command_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.rate_fps = self.specs['rate_fps']
        self.speed = self.specs['speed']
        self.turn = self.specs['turn']
        self.message = self.specs['message']

        self.moveBindings = self.specs['moveBindings'] if 'moveBindings' in self.specs.keys() else None
        self.topicBindings = self.specs['topicBindings'] if 'topicBindings' in self.specs.keys() else None
        if self.topicBindings is not None:
            self.publishers = {
                key: rospy.Publisher(
                    name=self.topicBindings[key],
                    data_class=Empty,
                    queue_size=10
                ) for key in self.topicBindings.keys()
            }
            cprint(f'topicBindings: \n {self.topicBindings}', self._logger)
        self.serviceBindings = None
        if 'serviceBindings' in self.specs.keys():
            self.serviceBindings = {}
            for service_specs in self.specs['serviceBindings']:
                self.serviceBindings[service_specs['key']] = {
                    'name': service_specs['name'],
                    'proxy': rospy.ServiceProxy(service_specs['name'], eval(service_specs['type'])),
                    'message': service_specs['message']
                }
            cprint(f'serviceBindings: \n {self.serviceBindings}', self._logger)
        self.methodBindings = self.specs['methodBindings'] if 'methodBindings' in self.specs.keys() else None
        if self.methodBindings is not None:
            self.camera_control_publisher = rospy.Publisher('/bebop/camera_control', Twist)
            for method in self.methodBindings.values():
                assert method in self.__dir__()  # make sure that method exists

        self.x = 0
        self.y = 0
        self.z = 0
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.reset_control_fields()

    def reset_control_fields(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.roll = 0
        self.pitch = 0
        self.yaw = 0

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key_name = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key_name

    def update_fields(self):
        key = self.get_key()
        if self.methodBindings is not None and key in self.methodBindings.keys():
            eval(f'self.{self.methodBindings[key]}()')
        if self.topicBindings is not None and key in self.topicBindings.keys():
            self.publishers[key].publish(Empty())
            cprint(f'publish {self.publishers[key]}', self._logger)
        if self.serviceBindings is not None and key in self.serviceBindings.keys():
            # rospy.wait_for_service(self.serviceBindings[key]['name'])
            self.serviceBindings[key]['proxy'](self.serviceBindings[key]['message'])
            # self.serviceBindings[key]['proxy'](True)
            cprint(f'{self.serviceBindings[key]["proxy"]}({self.serviceBindings[key]["message"]})', self._logger)
        if self.moveBindings is not None and key in self.moveBindings.keys():
            self.x = self.moveBindings[key][0]
            self.y = self.moveBindings[key][1]
            self.z = self.moveBindings[key][2]
            self.roll = self.moveBindings[key][3]
            self.pitch = self.moveBindings[key][4]
            self.yaw = self.moveBindings[key][5]
        else:
            self.reset_control_fields()

        return key

    def run(self):
        try:
            cprint(self.message, self._logger)
            while True:
                if self.update_fields() == '\x03':
                    break
                twist = Twist()
                twist.linear.x = self.x * self.speed
                twist.linear.y = self.y * self.speed
                twist.linear.z = self.z * self.speed
                twist.angular.x = self.roll * self.turn
                twist.angular.y = self.pitch * self.turn
                twist.angular.z = self.yaw * self.turn
                self.command_pub.publish(twist)
                rospy.sleep(1. / self.rate_fps)
                cprint(f'{twist}', self._logger)
        except Exception as e:
            print(e)
        finally:
            self.command_pub.publish(Twist())
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

    def toggle_camera_forward_down(self):
        if self.camera_direction == 'straight':
            twist = Twist()
            twist.angular.y = -90
            self.camera_control_publisher.publish(twist)
            self.camera_direction = 'down'
        elif self.camera_direction == 'down':
            twist = Twist()
            twist.angular.y = -13
            self.camera_control_publisher.publish(twist)
            self.camera_direction = 'tilt'
        elif self.camera_direction == 'tilt':
            twist = Twist()
            twist.angular.y = 0
            self.camera_control_publisher.publish(twist)
            self.camera_direction = 'straight'
        cprint(f'placed camera {self.camera_direction}', self._logger)


if __name__ == "__main__":
    keyboard = KeyboardActor()
    keyboard.run()



