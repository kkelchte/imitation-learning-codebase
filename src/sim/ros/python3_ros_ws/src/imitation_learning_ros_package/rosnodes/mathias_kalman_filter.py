#!/usr/bin/python3.8
from copy import copy
from typing import Union, Tuple

from geometry_msgs.msg import PointStamped, TwistStamped, PoseStamped, Point
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R


from src.sim.ros.src.utils import get_time_diff, get_timestamp, to_ros_time, euler_from_quaternion, \
    rotation_from_quaternion, quaternion_from_euler


class KalmanFilter(object):

    def __init__(self, model):
        '''
        Asynchronous kalman filter to estimate position.
        '''

        # Assign model matrices
        self.A = model.A
        self.B = model.B
        self.C = model.C

        # keep track of velocity commands and their actual timings to calculate correction step
        self.cmd_list = []

        # rot: indicates world_rotated ==> world yaw following yaw drone.
        # tnext: at next control step = t + Ts
        # hat: estimated
        # tmeas: at time of last measurements
        self.x_hat_rot_tnext = np.zeros(shape=(10, 1))  # 1/b0 * (px,vx,ax,py,vy,ay,pz,vz,ptheta,vtheta)
        self.x_rot_tmeas = np.zeros(shape=(10, 1))  # at last measurement time

        self.error_cov_hat_tmeas = np.zeros(10)  # error covariance matrix at the start?
        self.error_cov_hat_tnext = np.zeros(10)

        self.y_hat_rot_tmeas = np.zeros(shape=(8, 1))
        self.y_hat_rot_tnext = np.zeros(shape=(8, 1))
        self.prev_measurement = None
        self.tnext = None  # field keeping track of next control step time
        self.tmeas = None
        self.correcting = False  # field to avoid several correction steps running at the same time

        # Kalman tuning parameters.
        self.meas_noise_cov = np.identity(8)  # measurement noise covariance
        self.process_noise_cov = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # process noise covariance matrix
                                           [0, 1e1, 0, 0, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 1e1, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 1e1, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1e1]])
        self.requires_initialisation = True

    def reset(self):
        self.x_hat_rot_tnext = np.zeros(shape=(10, 1))  # 1/b0 * (px,vx,ax,py,vy,ay,pz,vz,ptheta,vtheta)
        self.x_rot_tmeas = np.zeros(shape=(10, 1))

        self.error_cov_hat_tmeas = np.zeros(10)
        self.error_cov_hat_tnext = np.zeros(10)

        self.y_hat_rot_tmeas = np.zeros(shape=(8, 1))
        self.y_hat_rot_tnext = np.zeros(shape=(8, 1))
        self.prev_measurement = None
        self.tnext = None
        self.tmeas = None

        # Kalman tuning parameters.
        self.meas_noise_cov = np.identity(8)  # measurement noise covariance
        self.process_noise_cov = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # process noise covariance matrix
                                           [0, 1e1, 0, 0, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 1e1, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 1e1, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1e1]])
        self.requires_initialisation = True

    def initialise(self, measurement: Odometry):
        self.prev_measurement = measurement
        self.tmeas = float(measurement.header.stamp.to_sec())
        zero_input_cmd = TwistStamped()
        zero_input_cmd.header = measurement.header
        self.cmd_list = [zero_input_cmd]
        self.requires_initialisation = False

    def kalman_prediction(self, input_cmd: TwistStamped, time_delay_s: float) -> Union[None, Odometry]:
        """
        Based on the velocity commands send out by the velocity controller,
        calculate a prediction of the position in the future.
        Arguments:
            input_cmd: TwistStamped
            time_delay_s: period of control updates Ts so pose and velocity at t+Ts are predicted based on input_cmd
        Returns:
            predicted pose as PointStamped
            velocity as PointStamped commands in world_yaw frame
        """
        if self.requires_initialisation:
            return None
        # keep input command for recalculation during correction step
        self.cmd_list.append(input_cmd)

        # general update
        self.x_hat_rot_tnext, self.y_hat_rot_tnext, self.error_cov_hat_tnext, self.tnext = self._predict_step_calc(
            input_cmd, time_delay_s, self.x_hat_rot_tnext, self.error_cov_hat_tnext)
        return self._generate_twist("tnext")

    def kalman_correction(self, measurement: Odometry,
                          time_delay_s: float) -> Union[None, Odometry]:
        """
        Whenever a new position measurement is available, sends this
        information to the perception and then triggers the kalman filter
        to apply a correction step.

        Arguments:
            measurement: Odometry expressed in "world" frame.
            time_delay_s: the period of the control updates

        Returns:
            Odometry estimate at tnext, unless there has been no prediction step, then tmeas.
            In case there is a kalman correction step already going on, ignore this measurement and return None.
        """
        if self.correcting:
            return None
        self.correcting = True
        if self.requires_initialisation:
            self.initialise(measurement)
        else:
            x_hat_rot_tmeas, error_cov_tmeas, last_command = self._predict_from_prev_meas_till_new_meas(measurement)
            self.x_rot_tmeas, self.y_hat_rot_tmeas, self.error_cov_hat_tmeas = self._correct_step_calc(measurement,
                                                                                                       x_hat_rot_tmeas,
                                                                                                       error_cov_tmeas)
            self.prev_measurement = measurement
            self.tmeas = float(measurement.header.stamp.to_sec())
            self._predict_from_new_meas_till_tnext(last_command, time_delay_s)
        twist = self._generate_twist("tnext" if self.tnext is not None else self.tmeas)
        self.correcting = False
        return twist

    def _generate_twist(self, time: str = "tnext") -> Odometry:
        """
        Adapt field y with observable system outputs to odometry message with correct time stamp
        with pose expressed in global frame and velocity expressed in rotated global frame according to the drone's yaw.
        """
        result = Odometry()
        result.header.stamp = to_ros_time(self.tnext if time == "tnext" else self.tmeas)
        result.header.frame_id = "global"
        data_vector = self.y_hat_rot_tnext if time == "tnext" else self.y_hat_rot_tmeas
        # transform y_hat_rot to y_hat based on estimated yaw
        yaw = data_vector[3, 0]
        position_rot = data_vector[0:3, 0]
        orientation = R.from_euler('XYZ', (0, 0, yaw), degrees=False)
        rotated_to_global = orientation.as_matrix()
        position = np.matmul(rotated_to_global, position_rot)

        result.pose.pose.position.x = position[0]
        result.pose.pose.position.y = position[1]
        result.pose.pose.position.z = position[2]
        result.pose.pose.orientation.x = orientation.as_quat()[0]
        result.pose.pose.orientation.y = orientation.as_quat()[1]
        result.pose.pose.orientation.z = orientation.as_quat()[2]
        result.pose.pose.orientation.w = orientation.as_quat()[3]

        result.twist.twist.linear.x = data_vector[4, 0]
        result.twist.twist.linear.y = data_vector[5, 0]
        result.twist.twist.linear.z = data_vector[6, 0]
        result.twist.twist.angular.z = data_vector[7, 0]

        return result

    def _predict_step_calc(self, input_cmd_stamped: TwistStamped, time_delay_s: float, x_hat_rot: np.ndarray,
                           error_cov_hat: np.ndarray, tstart: float = None) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Prediction step of the kalman filter. Update the position of the drone
        using the reference velocity commands.
        Arguments:
            - input_cmd_stamped = TwistStamped of command applied at t
            - Ts = varying step size over which to integrate.
            - x_hat_rot = current state as array 8x1
            - error_cov_hat = error covariance matrix a priori update
            - tstart = t for which x_hat_rot and error_cov_hat are predicted
        Returns:
            - x_hat_rot: X state estimate at time t+Ts
            - y_hat_rot: outputs y at time t+Ts
            - error_cov_hat: error covariance matrix at time t+Ts
            - tnext: end time t+Ts
        """
        u = np.array([[input_cmd_stamped.twist.linear.x],
                      [input_cmd_stamped.twist.linear.y],
                      [input_cmd_stamped.twist.linear.z],
                      [input_cmd_stamped.twist.angular.z]])
        # clip time period at zero so a negative delay leads to no change.
        time_delay_s = max(0., time_delay_s)
        x_hat_rot = np.matmul(time_delay_s * self.A + np.identity(10), x_hat_rot) + np.matmul(time_delay_s * self.B, u)
        y_hat_rot = np.matmul(self.C, x_hat_rot)

        error_cov_hat = np.matmul(time_delay_s * self.A + np.identity(10),
                                  np.matmul(error_cov_hat, np.transpose(time_delay_s * self.A + np.identity(10)))) \
            + self.process_noise_cov
        end_time = (get_timestamp(input_cmd_stamped) if tstart is None else tstart) + time_delay_s
        return x_hat_rot, y_hat_rot, error_cov_hat, end_time

    def _predict_from_prev_meas_till_new_meas(self, measurement):
        """
        Arguments:
            measurement: PoseStamped expressed in "world" frame, only used for timing
        Returns:
            x_hat_r: estimate state at measurement's time in rotated frame,
            error_cov: estimate error covariance at measurement's time,
            cmd: last command before measurement time arrived
        """
        # initialise local variables:
        x_hat_r = copy(self.x_rot_tmeas)
        error_cov = copy(self.error_cov_hat_tmeas)
        tnext = self.tmeas

        # discard irrelevant commands by ensuring only 1 command comes before previous measurement
        while len(self.cmd_list) > 1 and get_time_diff(self.prev_measurement, self.cmd_list[1]) > 0:
            self.cmd_list.pop(0)

        # predict from previous t_meas till first following command step using command applied before
        cmd_before = self.cmd_list.pop(0)
        if len(self.cmd_list) > 0:
            cmd_after = self.cmd_list.pop(0)
            time_difference = get_time_diff(cmd_after, self.prev_measurement)
            x_hat_r, y_hat_r, error_cov, tnext = self._predict_step_calc(cmd_before, time_difference, x_hat_r,
                                                                         error_cov, tnext)

            # predict for each following control up until one control step before t_meas
            # is there still a command in command list that is before t measurement, if so, apply it.
            one_more_step = len(self.cmd_list) != 0 and get_time_diff(measurement, self.cmd_list[0]) > 0
            while one_more_step:
                cmd_before = copy(cmd_after)
                cmd_after = self.cmd_list.pop(0)  # we know this command is still before t_measurement
                # use actually applied durations or clip by measurement time
                duration = get_time_diff(cmd_after, cmd_before)
                x_hat_r, y_hat_r, error_cov, tnext = self._predict_step_calc(cmd_before, duration,
                                                                             x_hat_r, error_cov, tnext)
                one_more_step = len(self.cmd_list) != 0 and get_time_diff(measurement, self.cmd_list[0]) > 0
            cmd_before = copy(cmd_after)

        # predict from control step till current measurement time using last control before measurement
        duration = get_timestamp(measurement) - tnext
        x_hat_r, y_hat_r, error_cov, tnext = self._predict_step_calc(cmd_before, duration,
                                                                     x_hat_r, error_cov, tnext)
        # prediction should be very close to measurement
        # assert abs(tnext - get_timestamp(measurement)) < 1e-4
        return x_hat_r, error_cov, cmd_before

    def _predict_from_new_meas_till_tnext(self, last_command: TwistStamped, time_delay_s: float):
        """
        Starting from x_rot_tmeas en error_cov_hat_tmeas at measurement time,
        continue last command till either next command start or for control_time_period.
        If there are already next commands available, update also for them the prediction.
        """
        # initialize local variables
        x_hat_r, error_cov, tnext = self.x_rot_tmeas, self.error_cov_hat_tmeas, self.tmeas
        # predict from t_meas till first following command step using command applied before
        cmds_to_keep = [last_command]
        cmd_before = last_command

        # predict from tmeas to first following control step using last_command
        if len(self.cmd_list) > 0:
            duration = get_time_diff(self.cmd_list[0], self.prev_measurement)
        else:
            duration = time_delay_s - get_time_diff(self.prev_measurement, cmd_before)

        x_hat_r, y_hat_r, error_cov, tnext = self._predict_step_calc(cmd_before, duration, x_hat_r, error_cov, tnext)

        while self.tnext is not None and self.tnext - tnext > 1e-4:
            cmd_after = self.cmd_list.pop(0) if len(self.cmd_list) > 0 else None
            duration = get_timestamp(cmd_after) - tnext if cmd_after is not None else time_delay_s
            x_hat_r, y_hat_r, error_cov, tnext = self._predict_step_calc(cmd_before, duration, x_hat_r, error_cov, tnext)
            if cmd_after is not None:
                cmd_before = copy(cmd_after)
                cmds_to_keep.append(cmd_after)

        self.x_hat_rot_tnext, self.y_hat_rot_tnext, self.error_cov_tnext, self.tnext = x_hat_r, \
                                                                                       y_hat_r, error_cov, tnext

        # reset cmd_list for next correction step with command before and after measurement
        self.cmd_list = cmds_to_keep

    def _correct_step_calc(self, measurement: Odometry, x_hat_rot_tmeas: np.ndarray, error_cov_hat_tmeas: np.ndarray):
        """
        Correction step of the kalman filter. Update the position of the drone in world_yaw frame
        using the measurements. Note that timings between measurment, state and covariance matrix should be aligned.
        Argument:
            - measurement = Odometry expressed in world frame.
            - x_hat_rot_tmeas = state at measurement time
            - error_cov_hat_tmeas = error covariance at measurement time
        Returns:

        """
        _, _, yaw = euler_from_quaternion((measurement.pose.pose.orientation.x,
                                           measurement.pose.pose.orientation.y,
                                           measurement.pose.pose.orientation.z,
                                           measurement.pose.pose.orientation.w))
        global_to_rotated = R.from_euler('XYZ', (0, 0, -yaw), degrees=False).as_matrix()

        position = np.array([[measurement.pose.pose.position.x],
                             [measurement.pose.pose.position.y],
                             [measurement.pose.pose.position.z]])
        position_rot = np.matmul(global_to_rotated, position)
        # assumption: velocity is expressed in global rotated frame => no need to rotate
        velocity_rot = np.array([[measurement.twist.twist.linear.x],
                                 [measurement.twist.twist.linear.y],
                                 [measurement.twist.twist.linear.z]])
        # output vector contains pose and velocity in global-rotated frame (following drone's yaw)
        y = np.zeros((8, 1))
        y[0:3] = position_rot
        y[3, 0] = yaw  # yaw is kept in global frame
        y[4:7] = velocity_rot
        y[7, 0] = measurement.twist.twist.angular.z  # angular y yaw velocity is kept in global frame

        # innovation = what is measured differently from what we would be expected
        # (note D is zero in y(i+1) = C x(i+1) + D j(i) )
        # only use pose to correct, don't correct velocity terms.
        nu = y - np.matmul(self.C, x_hat_rot_tmeas)

        # innovation_covariance depends on covariance of prediction Phat (reliability of prediction)
        # and measurement noise covariance R (reliability of measurement)
        innovation_covariance = np.matmul(self.C,
                                          np.matmul(error_cov_hat_tmeas, np.transpose(self.C))) + self.meas_noise_cov

        # Kalman gain represents impact of innovation on state estimate,
        # increases with covariance prediction (unreliable prediction)
        # decreases with covariance measurement error (unreliable measurement)
        kalman_gain = np.matmul(error_cov_hat_tmeas, np.matmul(np.transpose(self.C),
                                                               np.linalg.inv(innovation_covariance)))
        # update state
        x_hat_rot_tmeas = x_hat_rot_tmeas + np.matmul(kalman_gain, nu)
        # decrease state covariance
        error_cov_hat_tmeas = np.matmul((np.identity(10) - np.matmul(kalman_gain, self.C)), error_cov_hat_tmeas)
        # get pose and velocity information (note D is zero in y(i+1) = C x(i+1) + D j(i) )
        y_hat_rot_tnext = np.matmul(self.C, x_hat_rot_tmeas)

        return x_hat_rot_tmeas, y_hat_rot_tnext, error_cov_hat_tmeas
