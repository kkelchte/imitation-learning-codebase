import os
import time
from copy import deepcopy
from typing import Union
from sys import argv

import torch
import json
import h5py
import numpy as np
import subprocess
import shlex
import fgbg

from src.core.utils import get_date_time_tag
from src.core.data_types import TerminationType, Action
from src.sim.common.environment import EnvironmentConfig
from src.sim.ros.src.ros_environment import RosEnvironment
from src.sim.ros.src.utils import transform, set_random_cone_location, set_random_gate_location, spawn_line, \
    remove_line, send_reference_local

WORLD = 'gate_cone_line_realistic'
# TARGET = 'cone'  # cone gate line
TARGET = argv[1]
#DS_TASK = 'waypoints'  # 'velocities'  # waypoints
DS_TASK = argv[2]
NUMBER = 10
CHECKPOINT = os.path.join(os.environ["HOME"], 'code/contrastive-learning/data/best_down_stream', TARGET, DS_TASK)
# CHECKPOINT = os.path.join('/home/klaas/code/contrastive-learning/data/down_stream', DS_TASK, TARGET)
OUTPUTDIR = f"{os.environ['HOME']}/code/contrastive-learning/data/eval_online_ood/{DS_TASK}/{TARGET}"
os.makedirs(OUTPUTDIR, exist_ok=True)

print(f'{"x"*100}\n Running {NUMBER} times in world {WORLD} with target {TARGET} \n{"x"*100}')


def save(reference_pos,
         experience,
         json_data,
         hdf5_data,
         prediction) -> Union[float, None]:
    loss = None
    # only save in running state
    if experience.done != TerminationType.NotDone:
        return loss

    # store previous observation
    hdf5_data["observation"].append(deepcopy(experience.observation))

    # store velocity
    action = experience.action.value
    json_data["velocities"].append([action[0], action[1], action[2], action[5]])

    # store prediction
    if DS_TASK == 'waypoints':
        json_data["predictions"].append([float(p) for p in prediction])
    else:
        json_data["predictions"].append(
            [float(prediction.value[0]), float(prediction.value[1]), float(prediction.value[2]), float(prediction.value[5])]
        )

    # store relative target location
    relative_reference_position = transform(points=[np.asarray(reference_pos)],
                                            orientation=experience.info['position'][3:],
                                            translation=np.asarray(experience.info['position'][:3]),
                                            invert=True)[0]
    if TARGET == 'line':  # scale to fixed length
        norm = np.sqrt(np.sum(relative_reference_position ** 2))
        ref_distance = 0.5
        relative_reference_position = relative_reference_position * ref_distance / norm

    json_data["relative_target_location"].append(list(relative_reference_position))

    # store global target location
    json_data["global_target_location"].append(list(reference_pos))

    # store global drone location
    json_data["global_drone_pose"].append(list(experience.info['position']))

    # calculate loss
    if DS_TASK == 'velocities':
        output = prediction.value
        loss = np.sqrt(np.mean([(action[x] - output[x])**2 for x in [0, 1, 2, 5]]))
    else:
        loss = np.sqrt(np.mean([(relative_reference_position[x] - prediction[x])**2 for x in [0, 1, 2]]))
    json_data["rmse"].append(loss)
    return loss


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

    # store json data
    stored_data[episode_id] = json_data
    with open(output_json_path, "w") as f:
        json.dump(stored_data, f)

    # store hdf5 data
    hdf5_file = h5py.File(output_hdf5_path, "a")
    episode_group = hdf5_file.create_group(str(episode_id))
    for sensor_name in hdf5_data.keys():
        episode_group.create_dataset(
            sensor_name, data=np.stack(hdf5_data[sensor_name])
        )
    hdf5_file.close()

    # store trajectory
    os.makedirs(os.path.join(output_dir, 'trajectories'), exist_ok=True)
    fgbg.draw_trajectory(os.path.join(output_dir, 'trajectories', str(episode_id) + '.jpg'), 
                         json_data['global_target_location'][0], 
                         json_data['global_drone_pose'])
    
    return len(stored_data.keys())


if __name__ == '__main__':

    robot_dict = {
        'gate': "drone_sim_forward_cam",
        'cone': "drone_sim",
        'line': "drone_sim_down_cam"
    }

    environment_config_dict = {
        "output_path": OUTPUTDIR,
        "factory_key": "ROS",
        "max_number_of_steps": 150,
        "ros_config": {
            "observation": "camera",
            "info": ["position"],
            "action_topic": "/actor/mathias_controller/cmd_vel",
            "max_update_wait_period_s": 10,
            "visible_xterm": True,
            "step_rate_fps": 100,
            "ros_launch_config": {
                "random_seed": 123,
                "robot_name": robot_dict[TARGET],
                "fsm_mode": "TakeOverRun",
                "fsm": True,
                "altitude_control": True,
                "robot_display": False,
                "control_mapping": True,
                "waypoint_indicator": False,
                "control_mapping_config": "python" if DS_TASK == 'velocities' else 'mathias_controller',
                "world_name": WORLD,
                "starting_height": 1.5,
                "z_pos": 0.2,
                "yaw_or": 0.0,
                "gazebo": True,
            },
            "actor_configs": [{
                "name": "mathias_controller_with_KF",
                "file": f'{os.environ["CODEDIR"]}/src/sim/ros/config/actor/mathias_controller_with_KF.yml',
        }],
        }
    }
    config = EnvironmentConfig().create(
        config_dict=environment_config_dict
    )
    output_dir = config.output_path

    # Load model
    model = fgbg.DownstreamNet(output_size=(4,) if DS_TASK == 'velocities' else (3,))
    ckpt = torch.load(CHECKPOINT + '/checkpoint_model.ckpt', map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['state_dict'])
    print(f'Loaded encoder from {CHECKPOINT}.')

    environment = RosEnvironment(
        config=config
    )
    losses = []
    successes = []

    time.sleep(5)
    run_index = 0
    while run_index < NUMBER:
        # create output json and hdf5 file
        json_data = {
            "velocities": [],
            "global_target_location": [],
            "relative_target_location": [],
            "global_drone_pose": [],
            "predictions": [],
            "rmse": []
        }
        hdf5_data = {"observation": []}

        # collect info in corresponding objects
        if TARGET == 'cone':
            reference_pos = set_random_cone_location(WORLD)
        elif TARGET == 'gate':
            reference_pos = set_random_gate_location()
        elif TARGET == 'line':
            reference_pos = spawn_line(WORLD)
        else:
            raise NotImplementedError

        step_index = 0
        experience, observation = environment.reset()
        while experience.done == TerminationType.NotDone:
            tensor = torch.from_numpy(observation).permute(2, 0, 1).float().unsqueeze(0)
            output = model(tensor).detach().cpu().numpy().squeeze()
            if DS_TASK == 'waypoints':
                print(f'sending {output}')
                send_reference_local(output[0], output[1], output[2], delay=1)
            else:
                # output to action with velocities
                output = Action(value=np.asarray([output[0], output[1], output[2], 0, 0, output[3]]))
            experience, observation = environment.step(
                action=output if DS_TASK == 'velocities' else None)
            loss = save(reference_pos, experience, json_data, hdf5_data, prediction=output)
            if loss is not None:
                losses.append(loss)
            step_index += 1

        successes.append(experience.done == TerminationType.Success)
        if TARGET == 'line':
            remove_line()

        # dump data if run is successfully and at least 3 experiences are saved with something in view.
        print(f'{get_date_time_tag()}: stored {run_index + 1} / {NUMBER} with {sum(successes)} / {len(successes)} success')
        run_index = dump(json_data, hdf5_data, output_dir)

    # write results to output directory
    with open(os.path.join(config.output_path, 'results.txt'), 'w') as f:
        f.write(f'Success rate: {np.mean(successes)}\n')
        f.write(f'RMSE: {np.mean(losses)}\n')
        f.write(f'RMSE std: {np.std(losses)}\n')
    print(f'finished {environment.remove()}')
