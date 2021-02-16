#!/usr/bin/env python3.8

import numpy as np


class BebopModel(object):

    def __init__(self):
        '''Initializes the model to be used in the Kalman filter.
        State space model x'(t) = A*x(t) + B*u(t) in observable canonical
        form, corresponding to transfer function
                        b0
        G(s) = --------------------------
                s^3 + a2*s^2 + a1*s + a0
        State space model matrices for position Kalman filter are in
        continuous time. Are then converted to discrete time further on
        depending on varying Ts.
        '''

        # Building continuous A matrix
        a2x = 6.167
        a1x = 1.523
        a0x = 0.0
        Ax = np.array([[0., 1., 0.],
                       [0., 0., 1.],
                       [-a0x, -a1x, -a2x]])
        a2y = 5.147
        a1y = 2.116
        a0y = 0.0
        Ay = np.array([[0., 1., 0.],
                       [0., 0., 1.],
                       [-a0y, -a1y, -a2y]])
        a1z = 6.26
        a0z = 0.0
        Az = np.array([[0., 1.],
                       [-a0z, -a1z]])

        a1theta = 3.262
        a0theta = 0.0
        Atheta = np.array([[0., 1.],
                           [-a0theta, -a1theta]])
        self.A = np.zeros([10, 10])
        self.A[0:3, 0:3] = Ax
        self.A[3:6, 3:6] = Ay
        self.A[6:8, 6:8] = Az
        self.A[8:10, 8:10] = Atheta

        # continuous B matrix
        # control J or U is 4x1: linear x, linear y, linear z, angular z
        self.B = np.zeros([10, 4])
        self.B[2, 0] = 1
        self.B[5, 1] = 1
        self.B[7, 2] = 1
        self.B[9, 3] = 1

        # continuous C matrix
        b0x = 22.51  # b0 stems from transfer formula
        b0y = 18.96
        b0z = 6.066
        b0theta = 5.66

        self.C = np.zeros([8, 10])
        # pose x, y, z, yaw
        self.C[0, 0:3] = np.array([b0x, 0, 0])
        self.C[1, 3:6] = np.array([b0y, 0, 0])
        self.C[2, 6:8] = np.array([b0z, 0])
        self.C[3, 8:10] = np.array([b0theta, 0])

        # velocity x, y, z, yaw
        self.C[4, 0:3] = np.array([0, b0x, 0])
        self.C[5, 3:6] = np.array([0, b0y, 0])
        self.C[6, 6:8] = np.array([0, b0z])
        self.C[7, 8:10] = np.array([0, b0theta])

        # continuous D matrix is zero.
