import os
from sys import argv
import time
from copy import deepcopy

import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import rospy
import subprocess
import shlex
import cv2 as cv
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState

from src.sim.ros.src.process_wrappers import RosWrapper
from src.core.utils import get_date_time_tag
from src.core.data_types import TerminationType
from src.sim.common.environment import EnvironmentConfig
from src.sim.ros.src.ros_environment import RosEnvironment
from src.sim.ros.src.utils import transform, set_random_cone_location, set_random_gate_location, spawn_line, \
    spawn_flying_zone, remove_line, send_reference_global

WORLD = 'gate_cone_line_realistic'
#WORLD = 'gate_cone_line'
TARGET = 'red_line'
NUMBER = 30

print(f'{"x"*100}\n Running {NUMBER} times in world {WORLD} with target {TARGET} \n{"x"*100}')


def save(reference_pos,
         experience,
         json_data,
         hdf5_data,
         mask=None):
    # only save in running state
    if experience.done != TerminationType.NotDone:
        return

    if mask is None:
        # calculate mask
        mask = deepcopy(experience.observation)
        mask = np.mean(mask, axis=2)
        mask[mask == 1] = 0
        mask[mask != 0] = 1

    # don't save if item not in view
    if np.sum(mask) < (1000 if TARGET == 'gate' else 20):
        return

    # store mask
    hdf5_data["mask"].append(mask)

    # store previous observation
    hdf5_data["observation"].append(deepcopy(experience.observation))

    # store relative target location
    relative_reference_position = transform(points=[np.asarray(reference_pos)],
                                            orientation=experience.info['position'][3:],
                                            translation=np.asarray(experience.info['position'][:3]),
                                            invert=True)[0]
    if TARGET == 'line' or TARGET == 'red_line':  # scale to fixed length
        norm = np.sqrt(np.sum(relative_reference_position ** 2))
        ref_distance = 0.5
        relative_reference_position = relative_reference_position * ref_distance / norm

    json_data["relative_target_location"].append(list(relative_reference_position))

    # store global target location
    json_data["global_target_location"].append(list(reference_pos))

    # store global drone location
    json_data["global_drone_pose"].append(list(experience.info['position']))            


def dump(json_data, hdf5_data, output_dir):
    output_json_path = os.path.join(output_dir, 'data') + '.json'
    output_hdf5_path = os.path.join(output_dir, 'data') + '.hdf5'
    if os.path.isfile(output_json_path):
        with open(output_json_path, "r+") as f:
            stored_data = json.load(f)
    else:
        stored_data = {}

    if len(json_data["global_target_location"]) == 0:
        return len(stored_data.keys())

    episode_id = hash(tuple(json_data["global_target_location"][0]))

    stored_data[episode_id] = json_data
    with open(output_json_path, "w") as f:
        json.dump(stored_data, f)

    hdf5_file = h5py.File(output_hdf5_path, "a")
    episode_group = hdf5_file.create_group(str(episode_id))
    for sensor_name in hdf5_data.keys():
        episode_group.create_dataset(
            sensor_name, data=np.stack(hdf5_data[sensor_name])
        )
    hdf5_file.close()
    
    if False:  # WORLD == 'gate_cone_line_realistic':
        mask = np.asarray(hdf5_data['mask'][0]).squeeze()
        obs = np.asarray(hdf5_data['observation'][0]).squeeze()
        plt.imshow(np.stack([0.4 + mask] * 3, axis=-1) * obs)
        os.makedirs(os.path.join(output_dir, 'imgs'), exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'imgs', f'{episode_id}.jpg'))
        plt.close()
    return len(stored_data.keys())


def run_shortly_while_freezing_drone():
    # make drone's velocity zero but keep it at the same position
    get_model_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    old_model_state = get_model_state_service.call(model_name=rospy.get_param('/robot/model_name'))
    set_model_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    model_state = ModelState()
    model_state.pose = old_model_state.pose
    model_state.model_name = rospy.get_param('/robot/model_name')
    set_model_state_service.wait_for_service()
    set_model_state_service.call(model_state)

    # set fsm to overtake state so control mapper won't publish speed
    args = shlex.split("rostopic pub /fsm/overtake std_msgs/Empty '{}'")
    overtake_p = subprocess.Popen(args)

    # publish speed zero
    args = shlex.split("rostopic pub /cmd_vel geometry_msgs/Twist '{}'")
    speed_p = subprocess.Popen(args)

    overtake_p.terminate()
    speed_p.terminate()

if __name__ == '__main__':
    config = {
        'output_path': "static_camera_dataset",
        'gazebo': True,
        'robot_name': 'static_drone_sim_down_cam',
        'world_name': WORLD,
        'fsm': False,
        'control_mapping': False,
        'altitude_control': False,
        'mathias_controller_with_KF': False,
    }

    # spinoff roslaunch
    ros_process = RosWrapper(launch_file='load_ros.launch',
                             config=config,
                             visible=True)
    output_dir = config['output_path']
    
    run_index = 0
    while run_index < NUMBER:
        # create output json and hdf5 file
        json_data = {
            "relative_target_location": []}
        hdf5_data = {"mask": [],
                     "observation": []}

        # collect info in corresponding objects
        if TARGET == 'cone':
            reference_pos = set_random_cone_location(WORLD)
            send_reference_global(reference_pos[0], reference_pos[1], reference_pos[2])
        elif TARGET == 'gate':
            reference_pos = set_random_gate_location()
            send_reference_global(reference_pos[0], reference_pos[1], reference_pos[2])
        elif TARGET == 'line':
            reference_pos = spawn_line(WORLD)
            send_reference_global(reference_pos[0], reference_pos[1], reference_pos[2])
        elif TARGET == 'red_line':
            reference_pos = spawn_line(WORLD, color='red')
            send_reference_global(reference_pos[0], reference_pos[1], reference_pos[2])
        else:
            raise NotImplementedError

        # place camera somewhere randomly between in x, y, z sampled from [-0.5, +0.5]
        camera_position = np.random.uniform(-0.5, 0.5, 3)
        while not rospy.is_shutdown():
            time.sleep(1.)
