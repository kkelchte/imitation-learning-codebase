#!/usr/bin/python3.8

import rospy
from hector_uav_msgs.srv import EnableMotors
from std_msgs.msg import String, Empty
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from hector_uav_msgs.msg import *

from src.core.logger import get_logger, cprint
from src.sim.common.noise import *
from src.core.data_types import SensorType
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.src.utils import get_output_path, process_twist
from src.core.utils import camelcase_to_snake_format, get_filename_without_extension, safe_wait_till_true

"""
Take care of taking off and landing during take-over state
"""


class AltitudeControl:

    def __init__(self):
        self.count = 0
        rospy.init_node('altitude_control')
        safe_wait_till_true(f'kwargs["rospy"].has_param("/output_path")',
                            True, 5, 0.1, rospy=rospy)
        self._output_path = get_output_path()
        self._logger = get_logger(get_filename_without_extension(__file__), self._output_path)

        self._reference_height = rospy.get_param('/starting_height', 1)
        self._rate_fps = 60
        self._go_publisher = rospy.Publisher('/fsm/go', Empty, queue_size=10)
        self._robot = rospy.get_param('/robot/model_name', 'default')
        self._height = {}
        self._publishers = {}
        self._enable_motors_services = {}
        self._setup()

    def _setup(self):
        self._fsm_state = FsmState.Unknown
        rospy.Subscriber(name='/fsm/state',
                         data_class=String,
                         callback=self._set_fsm_state)

        # field turn True when motors are enabled, don't publish control when motors are disabled
        self._motors_enabled = False

        # keep track of last commands to detect stable point
        self._control_norm_window_length = 10
        self._control_norm_window = []

        if 'turtle' in self._robot or 'default' in self._robot or 'real' in self._robot:
            cprint(f'altitude control not required for {self._robot}, shutting down...', self._logger)
            sys.exit(0)
        elif self._robot == 'quadrotor':
            # in case of single quadrotor
            self._publishers['default'] = rospy.Publisher('cmd_vel', Twist, queue_size=10)
            sensor = SensorType.position
            sensor_topic = rospy.get_param(f'/robot/{sensor.name}_sensor/topic')
            sensor_type = rospy.get_param(f'/robot/{sensor.name}_sensor/type')
            rospy.Subscriber(name=sensor_topic,
                             data_class=eval(sensor_type),
                             callback=eval(f'self._process_{camelcase_to_snake_format(sensor_type)}'),
                             callback_args='default')
            rospy.wait_for_service('/enable_motors')
            self._enable_motors_services['default'] = rospy.ServiceProxy('/enable_motors', EnableMotors)
        elif isinstance(self._robot, list):
            # in case of tracking fleeing quadrotor
            self._publishers['tracking'] = rospy.Publisher('cmd_vel', Twist, queue_size=10)
            self._publishers['fleeing'] = rospy.Publisher('cmd_vel_1', Twist, queue_size=10)
            for agent in ['tracking', 'fleeing']:
                sensor = SensorType.position
                sensor_topic = rospy.get_param(f'/robot/{agent}_{sensor.name}_sensor/topic')
                sensor_type = rospy.get_param(f'/robot/{agent}_{sensor.name}_sensor/type')
                rospy.Subscriber(name=sensor_topic,
                                 data_class=eval(sensor_type),
                                 callback=eval(f'self._process_{camelcase_to_snake_format(sensor_type)}'),
                                 callback_args=agent)
                rospy.wait_for_service(f'/{agent}/enable_motors')
                self._enable_motors_services[agent] = rospy.ServiceProxy(f'/{agent}/enable_motors', EnableMotors)

        
    def _set_fsm_state(self, msg: String):
        # detect transition
        if self._fsm_state != FsmState[msg.data]:
            self._fsm_state = FsmState[msg.data]
            cprint(f'set state: {self._fsm_state}', self._logger)
            self._reference_height = rospy.get_param('/starting_height', 1)
            if self._fsm_state == FsmState.TakenOver:
                if not self._motors_enabled:
                    for agent, service in self._enable_motors_services.items():
                        cprint(f'starting motors {agent}', self._logger)
                        service.call(True)
                        rospy.sleep(3)
                        self._motors_enabled = True
            elif self._fsm_state == FsmState.Running:
                self._control_norm_window = []
                self._height = {}

    def _process_odometry(self, msg: Odometry, agent_name: str) -> None:
        if self._fsm_state == FsmState.TakenOver:
            self._height[agent_name] = msg.pose.pose.position.z

    def _process_pose_stamped(self, msg: PoseStamped, agent_name: str) -> None:
        if self._fsm_state == FsmState.TakenOver:
            self._height[agent_name] = msg.pose.position.z

    def _get_twist(self, agent: str = 'default') -> Twist:
        twist = Twist()
        height = self._height[agent]
        if height < (self._reference_height - 0.1):
            twist.linear.z = +0.5
        elif height > (self._reference_height + 0.1):
            twist.linear.z = -0.5
        else:
            twist.linear.z = 0
        return twist

    def run(self):
        rate = rospy.Rate(self._rate_fps)
        while not rospy.is_shutdown():
            if self._fsm_state == FsmState.TakenOver and self._motors_enabled:
                for agent, publisher in self._publishers.items():
                    if agent in self._height.keys():  # don't publish if we don't know the height yet
                        twist = self._get_twist(agent)
                        publisher.publish(twist)
                        self._control_norm_window.append(process_twist(twist).norm())
                        if len(self._control_norm_window) > self._control_norm_window_length:
                            self._control_norm_window.pop(0)
                if len(self._control_norm_window) == self._control_norm_window_length \
                        and np.mean(self._control_norm_window) < 0.5:
                    cprint(f'reached stable height', self._logger)
                    self._go_publisher.publish(Empty())
            rate.sleep()


if __name__ == "__main__":
    control = AltitudeControl()
    control.run()
