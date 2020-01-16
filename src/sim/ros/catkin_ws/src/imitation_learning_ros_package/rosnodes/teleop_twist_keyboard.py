#!/usr/bin/env python
from __future__ import print_function
import sys
import select
import termios
import tty
import yaml

import roslib
import rospy
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist

roslib.load_manifest('teleop_twist_keyboard')


def get_key():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key_name = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key_name


if __name__ == "__main__":
    settings = termios.tcgetattr(sys.stdin)
    rospy.init_node('teleop_twist_keyboard')

    command_topic = rospy.get_param("command_topic", "/cmd_vel")
    pub = rospy.Publisher(command_topic, Twist, queue_size=1)
    print("publishing on {0}".format(command_topic))

    speed = rospy.get_param("~speed", 0.5)
    turn = rospy.get_param("~turn", 1.0)
    x = 0
    y = 0
    z = 0
    th = 0
    status = 0

    config_file = rospy.get_param("keyboard_config")
    with open(config_file, 'r') as f:
        config = yaml.load(f)
    message = config['message']
    moveBindings = config['moveBindings']
    speedBindings = config['speedBindings']

    try:
        print(message)
        while True:
            key = get_key()
            if key in moveBindings.keys():
                x = moveBindings[key][0]
                y = moveBindings[key][1]
                z = moveBindings[key][2]
                th = moveBindings[key][3]
            elif key in speedBindings.keys():
                speed = speed * speedBindings[key][0]
                turn = turn * speedBindings[key][1]

                print("currently:\tspeed %s\tturn %s " % (speed, turn))
                if status == 14:
                    print(message)
                status = (status + 1) % 15
            else:
                x = 0
                y = 0
                z = 0
                th = 0
                if key == '\x03':
                    break

            twist = Twist()
            twist.linear.x = x*speed
            twist.linear.y = y*speed
            twist.linear.z = z*speed
            twist.angular.x = 0
            twist.angular.y = 0
            twist.angular.z = th*turn
            pub.publish(twist)

    except Exception as e:
        print(e)

    finally:
        twist = Twist()
        twist.linear.x = 0
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0
        pub.publish(twist)

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

