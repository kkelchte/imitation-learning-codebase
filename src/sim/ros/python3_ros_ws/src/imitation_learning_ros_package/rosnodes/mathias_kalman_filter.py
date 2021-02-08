#!/usr/bin/python3.8
from typing import Union

from geometry_msgs.msg import PointStamped, TwistStamped, PoseStamped, Point
import rospy
import numpy as np
from nav_msgs.msg import Odometry

from src.sim.ros.src.utils import get_time_diff


class KalmanFilter(object):

    def __init__(self, model, start_time):
        '''
        Asynchronous kalman filter to estimate position.
        '''

        # Assign model matrices
        self.A = model.A
        self.B = model.B
        self.C = model.C

        # keep track of velocity commands and their actual timings to calculate correction step

        command = TwistStamped()
        command.header.stamp = start_time
        self.cmd_list_after_prev_meas = [command]

        # rot: indicates world_rotated ==> world yaw following yaw drone.
        # tnext: at next control step = t + Ts
        # hat: estimated
        # tmeas: at time of last measurements
        self.x_hat_rot_tnext = np.zeros(shape=(8, 1))  # 1/b0 * (px,vx,ax,py,vy,ay,pz,vz) (TODO: ptheta, vtheta)
        self.x_rot_tmeas = np.zeros(shape=(8, 1))  # at last measurement time

        self.error_cov_hat_tmeas = np.zeros(8)  # error covariance matrix at the start?
        self.error_cov_hat_tnext = np.zeros(8)

        self.y_hat_rot_tmeas = PointStamped()

        # Kalman tuning parameters.
        self.meas_noise_cov = np.identity(3)  # measurement noise covariance
        self.process_noise_cov = np.array([[1, 0, 0, 0, 0, 0, 0, 0],  # process noise covariance matrix
                                           [0, 1e1, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 1, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 1, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 1e1, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 1, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 1, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 1e1]])

    def kalman_prediction(self, input_cmd: TwistStamped, time_delay_s: float):
        '''
        Based on the velocity commands send out by the velocity controller,
        calculate a prediction of the position in the future.
        Arguments:
            input_cmd: TwistStamped
            time_delay_s: period of control updates Ts so pose and velocity at t+Ts are predicted based on input_cmd
        Returns:
            predicted pose and velocity commands in world_yaw frame
        '''
        self.cmd_list_after_prev_meas.append(input_cmd)
        (self.x_hat_rot_tnext, yhat_r, vhat_r, self.error_cov_hat_tnext) = self._predict_step_calc(
            input_cmd, time_delay_s, self.x_hat_rot_tnext, self.error_cov_hat_tnext)
        return yhat_r, vhat_r

    def kalman_correction(self, measurement: Union[PoseStamped, TwistStamped],
                          time_delay_s: float):
        '''
        Whenever a new position measurement is available, sends this
        information to the perception and then triggers the kalman filter
        to apply a correction step.
        Cases:
            - case 1: multiple velocity commands between successive
                      measurements: ok
            - case 2: only one velocity command between successive
                      measurements: ok
            - case 3: no velocity command between successive measurements: ok
            - case 4: last velocity command corresponds to time after new
                      measurement, but due to delay is already in the list.
            - case 5: last velocity command before measurement is not yet in
                      the list and appears only in the next correction step. => not really possible
        Arguments:
            measurement: PoseStamped expressed in "world" frame.
            time_delay_s: the period of the control updates
        '''

        # discard last command before last measurement in case there are more commands
        # this command was only relevant if there were no other commands
        if (len(self.cmd_list_after_prev_meas) > 1) and \
                get_time_diff(self.y_hat_rot_tmeas, self.cmd_list_after_prev_meas[1]) > 0:
            self.cmd_list_after_prev_meas = self.cmd_list_after_prev_meas[1:]

        # First make prediction from old point t0 to last point t before new measurement.
        # Calculate variable B (time between latest prediction and new t0).
        # Check for case 4: last velocity command comes after measurement => don't use in prediction
        late_cmd_vel = []
        if get_time_diff(measurement, self.cmd_list_after_prev_meas[-1]) < 0:
            late_cmd_vel = [self.cmd_list_after_prev_meas[-1]]
            self.cmd_list_after_prev_meas = self.cmd_list_after_prev_meas[0:-1]
            print(f'KF: correction case 4')

        vel_len = len(self.cmd_list_after_prev_meas)
        if vel_len > 1:
            case3 = False  # get time from last measurement update till first command
            Ts = get_time_diff(self.cmd_list_after_prev_meas[1], self.y_hat_rot_tmeas)
        else:
            case3 = True  # get time from last measurement till now
            Ts = get_time_diff(measurement, self.y_hat_rot_tmeas)  # yhat_r_t0 = last measurement update in rotated frame
            print(f'KF: correction case 3')
        # kalman first predict step till t_meas' starting from t0 = t_meas
        (X, yhat_r, vhat_r, Phat) = self._predict_step_calc(
                self.cmd_list_after_prev_meas[0], Ts, self.x_rot_tmeas, self.error_cov_hat_tmeas)

        # If not case 2 or 3 -> need to predict up to
        # last vel cmd before new_t0
        if vel_len > 2:
            # Case 1.
            print(f'KF: correction case 1')
            for i in range(vel_len - 2):
                Ts = get_time_diff(
                    self.cmd_list_after_prev_meas[i+2], self.cmd_list_after_prev_meas[i+1])
                # print '\n kalman second predict step Ts and yhat_r \n', Ts, yhat_r.point
                (X, yhat_r, vhat_r, Phat) = self._predict_step_calc(
                    self.cmd_list_after_prev_meas[i+1], Ts, X, Phat)

        B = get_time_diff(measurement, self.cmd_list_after_prev_meas[-1])
        case5 = B > time_delay_s
        if case5:
            print(f'KF: correction case 5')

        # Now make prediction up to new t0 if not case 3.
        if not case3:
            # print '\n kalman third predict step Ts and yhat_r \n', B, yhat_r.point
            (X, yhat_r, vhat_r, Phat) = self._predict_step_calc(self.cmd_list_after_prev_meas[-1], B, X, Phat)
        else:
            case5 = False
            B = B % time_delay_s
            
        # Make a prediction estimate X for the time of the measurement based on previous measurement and commands
        # This prediction step depends on the number of control commands send between the last measurement and now
        # The control commands are stored in cmd_list_after_prev_meas.
        # If this list is empty, predict from last command before previous measurement step (case 3)

        # ---- CORRECTION ----
        # Correct the estimate at new t0 with the measurement.
        (X, self.y_hat_rot_tmeas, Phat) = self._correct_step_calc(measurement, X, Phat)
        self.x_rot_tmeas = X
        self.y_hat_rot_tmeas.header.stamp = measurement.header.stamp

        # Now predict until next point t that coincides with next time point for the controller.
        (X, yhat_r, vhat_r, Phat) = self._predict_step_calc(
                                self.cmd_list_after_prev_meas[-1],
                                (1 + case5)*time_delay_s - B,
                                X, Phat)  # ? Shouldn't we use the late_cmd_vel here as well ?

        # Save variable globally
        self.x_hat_rot_tnext = X
        self.error_cov_hat_tnext = Phat

        # Empty commands and only keep last
        self.cmd_list_after_prev_meas = [self.cmd_list_after_prev_meas[-1]]
        self.cmd_list_after_prev_meas += late_cmd_vel
        
        return yhat_r, self.y_hat_rot_tmeas

    def _predict_step_calc(self, input_cmd_stamped: TwistStamped,
                           Ts: float, X: np.ndarray, Phat: np.ndarray):
        """
        Prediction step of the kalman filter. Update the position of the drone
        using the reference velocity commands.
        Arguments:
            - input_cmd_stamped = TwistStamped of command applied at t
            - Ts = varying step size over which to integrate.
            - X = current state as array 8x1
            - Phat = error covariance matrix a priori update
        Returns:
            - X state estimate at time t+Ts
            - y,v at time t+Ts
            - Phat at time t+Ts
        """
        input_cmd = input_cmd_stamped.twist

        u = np.array([[input_cmd.linear.x],
                      [input_cmd.linear.y],
                      [input_cmd.linear.z]])
        # 8x8 * 8x1 + 8x3 * 3x1
        X = (np.matmul(Ts*self.A + np.identity(8), X) + np.matmul(Ts*self.B, u))

        Y = np.matmul(self.C, X)

        yhat_r = PointStamped()
        yhat_r.header.frame_id = "world_rot"
        yhat_r.point.x = Y[0, 0]
        yhat_r.point.y = Y[1, 0]
        yhat_r.point.z = Y[2, 0]

        vhat_r = PointStamped()
        vhat_r.header.frame_id = "world_rot"
        vhat_r.point.x = Y[3, 0]
        vhat_r.point.y = Y[4, 0]
        vhat_r.point.z = Y[5, 0]

        Phat = np.matmul(Ts*self.A + np.identity(8), np.matmul(
            Phat, np.transpose(Ts*self.A + np.identity(8)))) + self.process_noise_cov

        return X, yhat_r, vhat_r, Phat

    def _correct_step_calc(self, measurement: Union[Odometry, PoseStamped],
                           X: np.ndarray, Phat: np.ndarray):
        """
        Correction step of the kalman filter. Update the position of the drone in world_yaw frame
        using the measurements.
        Argument:
            - pos_meas = PoseStamped expressed in "world_yaw" frame.
            - X = "current" state
            - yhat_r = last
        """
        if isinstance(measurement, PoseStamped):
            y = np.array([[measurement.pose.position.x],
                          [measurement.pose.position.y],
                          [measurement.pose.position.z]])
        else:
            y = np.array([[measurement.pose.pose.position.x],
                          [measurement.pose.pose.position.y],
                          [measurement.pose.pose.position.z]])

        # TODO: make correction also on observed velocity in twist
        # innovation = what is measured differently from what we would be expected
        # (note D is zero in y(i+1) = C x(i+1) + D j(i) )
        # only use pose to correct, don't correct velocity terms.
        nu = y - np.matmul(self.C, X)[0:3, :]

        # covariance on innovation depends on covariance of prediction Phat (reliability of prediction)
        # and measurement noise covariance R (reliability of measurement)
        S = np.matmul(self.C[0:3, :], np.matmul(Phat, np.transpose(self.C[0:3, :]))) + self.meas_noise_cov

        # Kalman gain represents impact of innovation on state estimate,
        # increases with covariance prediction (unreliable prediction)
        # decreases with covariance measurement error (unreliable measurement)
        L = np.matmul(Phat, np.matmul(np.transpose(self.C[0:3, :]), np.linalg.inv(S)))
        # update state
        X = X + np.matmul(L, nu)
        # decrease state covariance
        Phat = np.matmul((np.identity(8) - np.matmul(L, self.C[0:3, :])), Phat)
        # store covariance of last measurement update (t0)
        self.error_cov_hat_tmeas = Phat
        # get pose and velocity information (note D is zero in y(i+1) = C x(i+1) + D j(i) )
        Y = np.matmul(self.C[0:3, :], X)

        yhat_r = PointStamped(point=Point(x=Y[0, 0],
                                          y=Y[1, 0],
                                          z=Y[2, 0]))
        return X, yhat_r, Phat

    # def transform_pose(self, pose, _from, _to):
    #     '''Transforms pose (geometry_msgs/PoseStamped) from frame "_from" to
    #     frame "_to".
    #     Arguments:
    #         - _from, _to = string, name of frame
    #     '''
    #     transform = self.get_transform(_from, _to)
    #     pose_tf = tf2_geom.do_transform_pose(pose, transform)
    #     pose_tf.header.stamp = pose.header.stamp
    #     pose_tf.header.frame_id = _to
    #
    #     return pose_tf

    # def get_transform(self, _from, _to):
    #     '''Returns the TransformStamped msg of the transform from reference
    #     frame '_from' to reference frame '_to'
    #     Arguments:
    #         - _from, _to = string, name of frame
    #     '''
    #     tf_f_in_t = self.tfBuffer.lookup_transform(
    #         _to, _from, rospy.Time(0), rospy.Duration(0.1))
    #     return tf_f_in_t
