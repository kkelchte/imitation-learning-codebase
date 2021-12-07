import os
from sys import argv
import shutil
import time
import unittest
from copy import deepcopy
from typing import Union

import h5py
import json
import torch
from std_msgs.msg import Empty
import numpy as np
import rospy
from geometry_msgs.msg import Twist, PointStamped, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32MultiArray, Empty
from sensor_msgs.msg import Image
from scipy.interpolate import interp1d
import fgbg

from src.core.data_types import Action
from src.sim.ros.src.utils import process_image, adapt_action_to_twist, process_odometry, process_twist
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
        self._episode_id = -1
        self._reset()
        rospy.Subscriber(name=rospy.get_param(f'/robot/camera_sensor/topic'),
                         data_class=eval(rospy.get_param(f'/robot/camera_sensor/type')),
                         callback=self._camera_callback,
                         callback_args=('observation',{}))
        rospy.Subscriber(name='/mask',
                         data_class=Image,
                         callback=self._camera_callback,
                         callback_args=('mask',{}))
        # odometry
        if rospy.has_param('/robot/position_sensor'):
            rospy.Subscriber(name=rospy.get_param('/robot/position_sensor/topic'),
                            data_class=eval(rospy.get_param('/robot/position_sensor/type')),
                            callback=self._set_field,
                            callback_args=('odometry', {})
            )
        # reference_pose
        rospy.Subscriber(name='/reference_pose',
                         data_class=PointStamped,
                         callback=self._set_field,
                         callback_args=('reference_pose', {}))
        # Applied action
        if rospy.has_param('/robot/command_topic'):
            rospy.Subscriber(name=rospy.get_param('/robot/command_topic'),
                             data_class=Twist,
                             callback=self._set_field,
                             callback_args=('action', {}))
        # fsm state
        rospy.Subscriber(name='/fsm/state',
                         data_class=String,
                         callback=self._set_field,
                         callback_args=('fsm_state', {}))
        self._fsm_state = FsmState.Unknown
        self._dumping = False
        

    def _dump(self):
        print('dumping')
        self._dumping = True

        output_hdf5_path = os.path.join(self.output_dir, 'data') + '.hdf5'
        hdf5_file = h5py.File(output_hdf5_path, "w")
        episode_group = hdf5_file.create_group(str(self._episode_id))
        for sensor_name in self._hdf5_data.keys():
            episode_group.create_dataset(
                sensor_name, data=np.stack(self._hdf5_data[sensor_name])
            )
        hdf5_file.close()

        output_json_path = os.path.join(self.output_dir, "data") + ".json"
        if os.path.isfile(output_json_path):
            with open(output_json_path, "r+") as f:
                stored_data = json.load(f)
        else:
            stored_data = {}
        print(f'stored episode {self._episode_id} in {self.output_dir}')
        # store json data
        stored_data[self._episode_id] = self._json_data
        with open(output_json_path, "w") as f:
            json.dump(stored_data, f)
        self._dumping = False



    def _reset(self):
        self._episode_id += 1
        self._hdf5_data = {"observation": [], "mask": []}
        self._json_data = {
            "velocities": [],
            "relative_target_location": [],
            "global_drone_pose": []
        }
        self._last_image = None
        self._last_mask = None
        self._last_odom = None
        self._last_reference_pose = None
        self._last_action = None

    def _set_field(self, msg: Union[Twist, Odometry, String], args: tuple) -> None:
        field_name, _ = args
        # print(f'set field {field_name} with {msg}')
        if field_name == 'fsm_state':
            dump = False
            if self._fsm_state == FsmState.Running and FsmState[msg.data] == FsmState.TakenOver:
                dump = True
            if self._fsm_state != FsmState[msg.data]:
                self._fsm_state = FsmState[msg.data]        
                print(f'set state to {self._fsm_state}')
            if dump:
                self._dump()
                self._reset()
        elif field_name == 'action':
            self._last_action = [float(p) for p in process_twist(msg).value]
        elif field_name == 'reference_pose':
            self._last_reference_pose = [msg.point.x, msg.point.y, msg.point.z]
        elif field_name == 'odometry':
            self._last_odom = [float(p) for p in process_odometry(msg)]

    def _camera_callback(self, msg: Image, args: tuple):
        # return
        field_name, _ = args
        if field_name == "observation":
            self._last_image = process_image(msg, {'height': 200, 'width': 200, 'depth': 3})
        if field_name == "mask":
            self._last_mask = process_image(msg, {'height': 200, 'width': 200, 'depth': 1})
            
    def _save(self):
        # store previous observation
        if self._dumping or self._fsm_state != FsmState.Running:
             return
        for a in [self._last_image, self._last_mask, self._last_odom, self._last_reference_pose, self._last_action]:
        # for a in [self._last_odom, self._last_reference_pose, self._last_action]:
            if a is None:
                return
        print(f'saving datapoint {len(self._json_data["velocities"])}')
        self._hdf5_data["observation"].append(self._last_image)
        self._hdf5_data["mask"].append(self._last_mask)
        self._json_data["global_drone_pose"].append(self._last_odom)
        self._json_data["relative_target_location"].append(self._last_reference_pose)
        self._json_data["velocities"].append(self._last_action)
        self._last_image = None
        self._last_mask = None
        self._last_odom = None
        self._last_action = None
        self._last_reference_pose = None

    def run(self):
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            rate.sleep()
            self._save()


if __name__ == '__main__':
    output_directory = f'{os.environ["HOME"]}/code/imitation-learning-codebase/experimental_data/real_world/droneroom_with_images/deep_supervision_triplet'
    # output_directory = f'{os.environ["HOME"]}/code/imitation-learning-codebase/experimental_data/real_world/droneroom/deep_supervision'
    # output_directory = f'{os.environ["HOME"]}/code/imitation-learning-codebase/experimental_data/real_world/droneroom/baseline'
    if os.path.isdir(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory, exist_ok=False)
    print(f"saving in {output_directory}")
    data_saver = Datasaver(output_dir=output_directory)
    data_saver.run()