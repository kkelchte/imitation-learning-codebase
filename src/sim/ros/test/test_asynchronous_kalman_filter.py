import os
import shutil
import time
import unittest

import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose, PointStamped, Point, TwistStamped, Twist, Vector3, PoseStamped, Quaternion
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty, Header
from std_srvs.srv import Empty as Emptyservice, EmptyRequest
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from src.core.utils import get_filename_without_extension, get_to_root_dir, get_data_dir, safe_wait_till_true
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.mathias_bebop_model import BebopModel
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.mathias_kalman_filter import KalmanFilter
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.src.utils import euler_from_quaternion, transform
from src.sim.ros.test.common_utils import TopicConfig, TestPublisherSubscriber, get_fake_laser_scan


def print_state(filter, pose=None, velocity=None):
    if False:
        x = filter.x_hat_rot_tnext
        print(f'px/b0: {x[0]}, vx/b0: {x[1]}, ax/b0: {x[2]}, '
              f'py/b0: {x[3]}, vy/b0: {x[4]}, ay/b0: {x[5]}, '
              f'pz/b0: {x[6]}, vz/b0: {x[7]}')
    if True:
        print(f'P_hat: {filter.error_cov_hat_tnext.sum()}')
    if True and pose is not None:
        print(f'pose: {pose.point.z}')
    if True and velocity is not None:
        print(f'velocity: {velocity.point.z}')


class TestAsynchronousKalmanFilter(unittest.TestCase):

    @unittest.skip
    def test_prediction_step(self):

        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

        config = {
            'output_path': self.output_dir,
            'gazebo': False,
            'fsm': False,
            'control_mapping': False,
            'ros_expert': False,
        }
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=config,
                                       visible=True)
        rospy.init_node('testKF')

        model = BebopModel()

        # initialize filter with zero command
        filter = KalmanFilter(model=model,
                              start_time=rospy.Time.now())

        # place initial measurement with expected control rate of 10Hz
        measurement = PoseStamped()
        measurement.header.stamp = rospy.Time.now()
        y, v = filter.kalman_correction(measurement, time_delay_s=0.1)

        print_state(filter, y, v)

        print('5x get prediction of next state based on upwards command')
        # see that z increases and P
        init_p = np.sum(filter.error_cov_hat_tnext)
        init_z = y.point.z
        for _ in range(5):
            rospy.sleep(0.1)
            command = TwistStamped(twist=Twist(linear=Vector3(x=0, y=0, z=1)))
            command.header.stamp = rospy.Time.now()
            y, v = filter.kalman_prediction(command, time_delay_s=0.1)
        self.assertGreater(y.point.z, init_z)
        self.assertGreater(np.sum(filter.error_cov_hat_tnext), init_p)

        print_state(filter, y, v)

        print('correct with plausible pose measurement: case 1')
        init_p = np.sum(filter.error_cov_hat_tnext)
        measurement = PoseStamped(pose=Pose(position=Point(x=0, y=0, z=0.5)))
        measurement.header.stamp = rospy.Time.now()
        y, v = filter.kalman_correction(measurement, time_delay_s=0.1)
        print_state(filter, y, v)
        # covariance should decrease (if proposed measurement is small enough)
        self.assertGreater(init_p, np.sum(filter.error_cov_hat_tnext))

        print('apply prediction 1 step in the future based on command')
        command = TwistStamped(twist=Twist(linear=Vector3(x=0, y=0, z=1)))
        command.header.stamp = rospy.Time.now()
        y, v = filter.kalman_prediction(command, time_delay_s=0.1)
        rospy.sleep(0.09)
        print_state(filter, y, v)

        print('correct with plausible pose measurement: case 2')
        measurement = PoseStamped(pose=Pose(position=Point(x=0, y=0, z=1.5)))
        measurement.header.stamp = rospy.Time.now()
        y, v = filter.kalman_correction(measurement, time_delay_s=0.1)
        rospy.sleep(0.09)
        print_state(filter, y, v)

        print('correct with plausible pose measurement: case 3')
        measurement = PoseStamped(pose=Pose(position=Point(x=0, y=0, z=0.5)))
        measurement.header.stamp = rospy.Time.now()
        y, v = filter.kalman_correction(measurement, time_delay_s=0.1)
        rospy.sleep(0.09)
        print_state(filter, y, v)

        print('Case 4: apply prediction 1 step in the future based on command')
        print('correct with plausible pose measurement occurring before previous command: case 4')

        first_stamp = rospy.Time.now()
        rospy.sleep(0.1)
        second_stamp = rospy.Time.now()

        command = TwistStamped(twist=Twist(linear=Vector3(x=0, y=0, z=1)))
        command.header.stamp = second_stamp
        y, v = filter.kalman_prediction(command, time_delay_s=0.1)
        print_state(filter, y, v)

        measurement = PoseStamped(pose=Pose(position=Point(x=0, y=0, z=0.5)))
        measurement.header.stamp = first_stamp
        y, v = filter.kalman_correction(measurement, time_delay_s=0.1)
        rospy.sleep(0.1)
        print_state(filter, y, v)

        print('correct with plausible pose measurement occurring after last command '
              'for which no prediction was done: case 5')
        rospy.sleep(0.2)
        measurement = PoseStamped(pose=Pose(position=Point(x=0, y=0, z=0.5)))
        measurement.header.stamp = first_stamp
        y, v = filter.kalman_correction(measurement, time_delay_s=0.1)
        print_state(filter, y, v)

    def test_idle_state_with_inf_corrections(self):

        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

        config = {
            'output_path': self.output_dir,
            'gazebo': False,
            'fsm': False,
            'control_mapping': False,
            'ros_expert': False,
        }
        self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                       config=config,
                                       visible=True)
        rospy.init_node('testKF')

        model = BebopModel()

        # initialize filter with zero command
        filter = KalmanFilter(model=model,
                              start_time=rospy.Time.now())
        print_state(filter)

        for _ in range(10):
            rospy.sleep(0.1)
            # place initial measurement with expected control rate of 10Hz
            measurement = PoseStamped(pose=Pose(position=Point(x=0, y=0, z=0.5)))
            measurement.header.stamp = rospy.Time.now()
            y, v = filter.kalman_correction(measurement, time_delay_s=0.1)

            print_state(filter, y, v)

    def tearDown(self) -> None:
        self._ros_process.terminate()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
