import os
import shutil
import time
import unittest

import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose, PointStamped, Point, TwistStamped, Twist, Vector3
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


def print_state(x):
    print(f'px/b0: {x[0]}, vx/b0: {x[1]}, ax/b0: {x[2]}, '
          f'py/b0: {x[3]}, vy/b0: {x[4]}, ay/b0: {x[5]}, '
          f'pz/b0: {x[6]}, vz/b0: {x[7]}')


class TestAsynchronousKalmanFilter(unittest.TestCase):

    def test_prediction_step(self):
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

        model = BebopModel()
        filter = KalmanFilter(model=model)
        init_state = filter.X_r[:]
        print_state(init_state)
        for _ in range(10):
            cmd = TwistStamped(twist=Twist(linear=Vector3(x=1, y=1, z=1)))
            yhat_r, vhat_r = filter.kalman_pos_predict(cmd)
            print_state(filter.X_r)
            print(f'Phat: {filter.Phat}, yhat_r: {yhat_r}, vhat_r: {vhat_r}')

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
