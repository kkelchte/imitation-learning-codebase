import os
import shutil
import time
import unittest

import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose, PointStamped, Point, TwistStamped, Twist, Vector3, PoseStamped, Quaternion, \
    PoseWithCovariance
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
from src.sim.ros.src.utils import euler_from_quaternion, transform, to_ros_time
from src.sim.ros.test.common_utils import TopicConfig, TestPublisherSubscriber, get_fake_laser_scan


def print_result(result: Odometry, tag: str = ''):
    if result is None:
        print('None')
    else:
        _, _, yaw = euler_from_quaternion((result.pose.pose.orientation.x,
                                           result.pose.pose.orientation.y,
                                           result.pose.pose.orientation.z,
                                           result.pose.pose.orientation.w))

        print(f'{tag}{result.header.stamp.to_sec()}: pose ({result.pose.pose.position.x}, '
              f'{result.pose.pose.position.y}, {result.pose.pose.position.z}, {yaw}), '
              f'velocity ({result.twist.twist.linear.x}, {result.twist.twist.linear.y}, {result.twist.twist.linear.z},'
              f'{result.twist.twist.angular.z})')


def print_state(filter, pose=None, velocity=None):
    if True:
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

    def test_normal_behavior(self):
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        self._ros_process = None

        model = BebopModel()
        filter = KalmanFilter(model=model)

        # send measurement 0
        t = 0
        measurement = Odometry(header=Header(stamp=to_ros_time(t)),
                               pose=PoseWithCovariance(pose=Pose(position=Point(z=0.))),
                               twist=TwistStamped(twist=Twist(linear=Point(z=0.5))))
        result = filter.kalman_correction(measurement, 2)
        print_result(result)
        self.assertTrue(filter.tnext is None)
        self.assertTrue(filter.tmeas == 0)
        self.assertTrue(len(filter.cmd_list) == 1)

        # predict control 0
        t = 1
        command = TwistStamped(Header(stamp=to_ros_time(t)), Twist(linear=Point(z=0.5)))
        result = filter.kalman_prediction(command, 2)
        print_result(result)
        self.assertTrue(filter.tnext == 3)
        self.assertTrue(filter.tmeas == 0)
        self.assertTrue(len(filter.cmd_list) == 2)

        # predict control 1
        t = 3
        command = TwistStamped(Header(stamp=to_ros_time(t)), Twist(linear=Point(z=0.5)))
        result = filter.kalman_prediction(command, 2)
        print_result(result)
        self.assertTrue(filter.tnext == 5)
        self.assertTrue(filter.tmeas == 0)
        self.assertTrue(len(filter.cmd_list) == 3)

        # send measurement 1
        t = 4
        measurement = Odometry(header=Header(stamp=to_ros_time(t)),
                               pose=PoseWithCovariance(pose=Pose(position=Point(z=20.))))
        result = filter.kalman_correction(measurement, 2)
        print_result(result)
        self.assertTrue(filter.tnext == 5)
        self.assertTrue(filter.tmeas == 4)
        self.assertTrue(len(filter.cmd_list) == 1)

        # predict control 2
        t = 5
        command = TwistStamped(Header(stamp=to_ros_time(t)), Twist(linear=Point(z=0.5)))
        result = filter.kalman_prediction(command, 2)
        print_result(result)
        self.assertTrue(filter.tnext == 7)
        self.assertTrue(filter.tmeas == 4)
        self.assertTrue(len(filter.cmd_list) == 2)

        # predict control 3
        t = 7
        command = TwistStamped(Header(stamp=to_ros_time(t)), Twist(linear=Point(z=0.5)))
        result = filter.kalman_prediction(command, 2)
        print_result(result)
        self.assertTrue(filter.tnext == 9)
        self.assertTrue(filter.tmeas == 4)
        self.assertTrue(len(filter.cmd_list) == 3)

        # send measurement 1
        t = 8
        measurement = Odometry(header=Header(stamp=to_ros_time(t)),
                               pose=PoseWithCovariance(pose=Pose(position=Point(z=20.))))
        result = filter.kalman_correction(measurement, 1)
        print_result(result)
        self.assertTrue(filter.tnext == 9)
        self.assertTrue(filter.tmeas == 8)
        self.assertTrue(len(filter.cmd_list) == 1)

    def realistic_period(self):
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        self._ros_process = None

        model = BebopModel()
        filter = KalmanFilter(model=model)

        measurement_rate = 5
        prediction_rate = 15
        data = {
            'predicted': [],
            'adjusted': [],
            'measured': []
        }

        # send measurement 0
        t = 0
        measurement = Odometry(header=Header(stamp=to_ros_time(t)),
                               pose=PoseWithCovariance(pose=Pose(position=Point(z=0.))),
                               twist=TwistStamped(twist=Twist(linear=Point(z=0.))))
        result = filter.kalman_correction(measurement, 1./prediction_rate)
        print_result(result, 'measurement: ')
        data['measured'].append((t, 0))
        data['adjusted'].append((t, result.pose.pose.position.x))

        # run for 3 seconds
        for _ in range(1, 3*prediction_rate):
            t = _ * 1/prediction_rate
            # predict control
            command = TwistStamped(Header(stamp=to_ros_time(t)), Twist(linear=Point(x=1.0)))
            result = filter.kalman_prediction(command, 1./prediction_rate)
            print_result(result, 'prediction: ')
            data['predicted'].append((filter.tnext, result.pose.pose.position.x))

            if _ % measurement_rate == 0:
                # predict measurement
                measurement = Odometry(header=Header(stamp=to_ros_time(t)),
                                       pose=PoseWithCovariance(pose=Pose(position=Point(x=0.1 * (_ // measurement_rate)))),
                                       twist=TwistStamped(twist=Twist(linear=Point(x=0.5 * (_ // measurement_rate)))))
                result = filter.kalman_correction(measurement, 1. / prediction_rate)
                print_result(result, 'measurement: ')
                data['measured'].append((t, 0.1 * (_ // measurement_rate)))
                data['adjusted'].append((filter.tmeas, result.pose.pose.position.x))

        return data

    def late_measurements_period(self):
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        self._ros_process = None

        model = BebopModel()
        filter = KalmanFilter(model=model)

        measurement_rate = 5
        prediction_rate = 15
        data = {
            'predicted': [],
            'adjusted': [],
            'measured': []
        }

        # send measurement 0
        t = 0
        measurement = Odometry(header=Header(stamp=to_ros_time(t)),
                               pose=PoseWithCovariance(pose=Pose(position=Point(z=0.))),
                               twist=TwistStamped(twist=Twist(linear=Point(z=0.))))
        result = filter.kalman_correction(measurement, 1./prediction_rate)
        print_result(result, 'measurement: ')
        data['measured'].append((t, 0))
        data['adjusted'].append((t, result.pose.pose.position.x))

        # run for 3 seconds
        for _ in range(1, 3*prediction_rate):
            t = _ * 1/prediction_rate
            # predict control
            command = TwistStamped(Header(stamp=to_ros_time(t)), Twist(linear=Point(x=1.0)))
            result = filter.kalman_prediction(command, 1./prediction_rate)
            print_result(result, 'prediction: ')
            data['predicted'].append((filter.tnext, result.pose.pose.position.x))

            if _ % measurement_rate == 0:
                # predict measurement
                measurement = Odometry(header=Header(stamp=to_ros_time(t - 0.01)),
                                       pose=PoseWithCovariance(pose=Pose(position=Point(x=0.1 * (_ // measurement_rate)))),
                                       twist=TwistStamped(twist=Twist(linear=Point(x=0.5 * (_ // measurement_rate)))))
                result = filter.kalman_correction(measurement, 1. / prediction_rate)
                print_result(result, 'measurement: ')
                data['measured'].append((t, 0.1 * (_ // measurement_rate)))
                data['adjusted'].append((filter.tmeas, result.pose.pose.position.x))

        return data

    def early_measurements_period(self):
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        self._ros_process = None

        model = BebopModel()
        filter = KalmanFilter(model=model)

        measurement_rate = 5
        prediction_rate = 15
        data = {
            'predicted': [],
            'adjusted': [],
            'measured': []
        }

        # send measurement 0
        t = 0
        measurement = Odometry(header=Header(stamp=to_ros_time(t)),
                               pose=PoseWithCovariance(pose=Pose(position=Point(z=0.))),
                               twist=TwistStamped(twist=Twist(linear=Point(z=0.))))
        result = filter.kalman_correction(measurement, 1./prediction_rate)
        print_result(result, 'measurement: ')
        data['measured'].append((t, 0))
        data['adjusted'].append((t, result.pose.pose.position.x))

        # run for 3 seconds
        for _ in range(1, 3*prediction_rate):
            t = _ * 1/prediction_rate
            if _ % measurement_rate == 0:
                # predict measurement
                measurement = Odometry(header=Header(stamp=to_ros_time(t + 0.01)),
                                       pose=PoseWithCovariance(pose=Pose(position=Point(x=0.1 * (_ // measurement_rate)))),
                                       twist=TwistStamped(twist=Twist(linear=Point(x=0.5 * (_ // measurement_rate)))))
                result = filter.kalman_correction(measurement, 1. / prediction_rate)
                print_result(result, 'measurement: ')
                data['measured'].append((t, 0.1 * (_ // measurement_rate)))
                data['adjusted'].append((filter.tmeas, result.pose.pose.position.x))

            # predict control
            command = TwistStamped(Header(stamp=to_ros_time(t)), Twist(linear=Point(x=1.0)))
            result = filter.kalman_prediction(command, 1./prediction_rate)
            print_result(result, 'prediction: ')
            data['predicted'].append((filter.tnext, result.pose.pose.position.x))
        return data

    def test_time_glitches(self):
        markers = {'predicted': '.', 'adjusted': '+', 'measured': '1'}
        fig = plt.figure(figsize=(10, 10))
        name = 'normal'
        color = 'C0'
        data = self.realistic_period()
        for k in data.keys():
            plt.plot([d[0] for d in data[k]],
                     [d[1] for d in data[k]],
                     marker=markers[k],
                     color=color,
                     label=f'{name}: {k}')

        name = 'late'
        color = 'C1'
        data = self.late_measurements_period()
        for k in data.keys():
            plt.plot([d[0] for d in data[k]],
                     [d[1] for d in data[k]],
                     marker=markers[k],
                     color=color,
                     label=f'{name}: {k}')

        name = 'early'
        color = 'C2'
        data = self.early_measurements_period()
        for k in data.keys():
            plt.plot([d[0] for d in data[k]],
                     [d[1] for d in data[k]],
                     marker=markers[k],
                     color=color,
                     label=f'{name}: {k}')
        plt.legend()
        plt.show()

    def tearDown(self) -> None:
        if self._ros_process is not None:
            self._ros_process.terminate()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
