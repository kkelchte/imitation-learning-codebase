#!/usr/bin/python3.7
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

    command_topic = rospy.get_param("/actor/keyboard/command_topic")

    pub = rospy.Publisher(command_topic, Twist, queue_size=1)
    print("publishing on {0}".format(command_topic))

    rate_fps = rospy.get_param('/actor/keyboard/rate_fps', 20)
    speed = rospy.get_param("/actor/keyboard/speed", 0.5)
    turn = rospy.get_param("/actor/keyboard/turn", 1.0)
    x = 0
    y = 0
    z = 0
    th = 0
    status = 0

    # config_file = rospy.get_param("keyboard_config")
    # with open(config_file, 'r') as f:
    #     config = yaml.load(f)
    message = rospy.get_param('/actor/keyboard/message', '### Could not find config message.')
    moveBindings = rospy.get_param('/actor/keyboard/moveBindings')
    speedBindings = rospy.get_param('/actor/keyboard/speedBindings')
    topicBindings = rospy.get_param('/actor/keyboard/topicBindings')
    publishers = {
        key: rospy.Publisher(
                name=rospy.get_param(topicBindings[key]),
                data_class=Empty,
                queue_size=10
             ) for key in topicBindings.keys()
    }
    try:
        print(message)
        while True:
            key = get_key()
            if key in topicBindings.keys():
                publishers[key].publish(Empty())
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

            rospy.sleep(1./rate_fps)
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

