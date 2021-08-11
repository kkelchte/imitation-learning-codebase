import os
from sys import argv
import shutil
import time
import unittest
from copy import deepcopy
from typing import Union

import h5py
import torch
from std_msgs.msg import Empty
import numpy as np
import rospy
from geometry_msgs.msg import Twist, PointStamped, Point
from std_msgs.msg import String, Float32MultiArray, Empty
from sensor_msgs.msg import Image
from scipy.interpolate import interp1d
import fgbg

from src.core.data_types import Action
from src.sim.ros.src.utils import process_image, adapt_action_to_twist
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import Fsm, FsmState

class Datasaver:

    def __init__(self, output_dir):
        self.output_dir = output_dir
        rospy.init_node('datasaver')
        # wait till ROS started properly
        stime = time.time()
        max_duration = 60
        while not rospy.has_param('/output_path') and time.time() < stime + max_duration:
            time.sleep(0.01)
        rospy.Subscriber(name=rospy.get_param(f'/robot/camera_sensor/topic'),
                         data_class=eval(rospy.get_param(f'/robot/camera_sensor/type')),
                         callback=self._camera_callback)
        # fsm state
        rospy.Subscriber(name='/fsm/state',
                         data_class=String,
                         callback=self._set_fsm_state)
        self._fsm_state = FsmState.Unknown
        self._episode_id = -1
        self._reset()

    def _dump(self):
        print(f'stored movie in {self.output_dir}')
        output_hdf5_path = os.path.join(self.output_dir, 'data') + '.hdf5'
        hdf5_file = h5py.File(output_hdf5_path, "a")
        episode_group = hdf5_file.create_group(str(self._episode_id))
        for sensor_name in self._hdf5_data.keys():
            episode_group.create_dataset(
                sensor_name, data=np.stack(self._hdf5_data[sensor_name])
            )
        hdf5_file.close()        

    def _reset(self):
        self._episode_id += 1
        self._hdf5_data = {"observation": []}

    def _set_fsm_state(self, msg: String):
        if self._fsm_state == FsmState.Running and FsmState[msg.data] == FsmState.TakenOver:
            self._dump()
            self._reset()
        if self._fsm_state != FsmState[msg.data]:
            self._fsm_state = FsmState[msg.data]        
            print(f'set state to {self._fsm_state}')
            
    def _camera_callback(self, msg: Image):
        image = process_image(msg, {'height': 200, 'width': 200, 'depth': 3})
        if self._fsm_state == FsmState.Running:
            # store previous observation
            self._hdf5_data["observation"].append(deepcopy(image))

    def run(self):
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            rate.sleep()


if __name__ == '__main__':
    target = 'gate'
    output_directory = f'{os.environ["HOME"]}/code/contrastive-learning/data/datasets/bebop_real_movies/{target}'
    os.makedirs(output_directory, exist_ok=True)
    print(f"saving in {output_directory}")
    data_saver = Datasaver(output_dir=output_directory)
    data_saver.run()