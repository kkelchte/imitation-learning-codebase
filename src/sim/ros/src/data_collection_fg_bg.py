import os
import shutil
import time
import unittest
from copy import deepcopy

from tqdm import tqdm
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import rospy
import subprocess
import shlex
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
from geometry_msgs.msg import Pose
import xml.etree.ElementTree as ET
from scipy.interpolate import interp1d

from src.core.utils import get_filename_without_extension, get_to_root_dir, get_date_time_tag
from src.core.data_types import TerminationType, SensorType
from src.sim.common.environment import EnvironmentConfig
from src.sim.ros.src.ros_environment import RosEnvironment
from src.sim.ros.test.common_utils import TopicConfig, TestPublisherSubscriber
from src.sim.ros.src.utils import transform

WORLD = 'gate_cone_line'
TARGET = 'gate'
NUMBER = 100


def update_line_model():
    # create random line from x -4 till +4
    number_of_points = 70
    xmin = -1
    xmax = 2
    x = np.asarray([xmin, 0, xmax])
    y = np.asarray([np.random.uniform(-xmin, xmin), 0, np.random.uniform(-xmax, xmax)])

    interpolation = interp1d(x, y, kind='quadratic')
    x_coords = np.linspace(xmin, xmax, num=number_of_points, endpoint=True)
    y_coords = interpolation(x_coords)

    # extract waypoint and goal
    goal_x_est = 1.75
    idx = (np.abs(x_coords - goal_x_est)).argmin()
    x_g, y_g = x_coords[idx], y_coords[idx]
    z_g = np.random.uniform(0.7, 1.7)
    send_reference_and_set_goal(x_g, y_g, z_g)

    # create a world file with corresponding tubes
    r = 0.01
    l = 0.06

    # Load line_segment:
    model_dir = 'src/sim/ros/gazebo/models/line_segment'
    tree = ET.parse(os.path.join(os.environ['PWD'], model_dir, 'line.sdf'))
    root = tree.getroot()

    # remove any existing model
    for child in root:
        if child.get('name') == 'line':
            root.remove(child)

    # add model to world
    model = ET.SubElement(root, 'model', attrib={'name': 'line'})
    # Place small cylinders in one model
    for index, (x, y) in enumerate(zip(x_coords, y_coords)):
        static = ET.SubElement(model, 'static')
        static.text = '1'
        link = ET.SubElement(model, 'link', attrib={'name': f'link_{index}'})
        pose = ET.SubElement(link, 'pose', attrib={'frame': ''})
        next_x = x_coords[(index + 1) % len(x_coords)]
        next_y = y_coords[(index + 1) % len(x_coords)]
        derivative = (next_y - y) / (next_x - x)
        slope = np.arctan(derivative)
        pose.text = f'{x} {y} {r} 0 1.57 {slope}'
        collision = ET.SubElement(link, 'collision', attrib={'name': 'collision'})
        visual = ET.SubElement(link, 'visual', attrib={'name': 'visual'})
        material = ET.SubElement(visual, 'material')
        script = ET.SubElement(material, 'script')
        name = ET.SubElement(script, 'name')
        name.text = 'Gazebo/Black'
        uri = ET.SubElement(script, 'uri')
        uri.text = 'file://media/materials/scripts/gazebo.material'
        for element in [collision, visual]:
            geo = ET.SubElement(element, 'geometry')
            cylinder = ET.SubElement(geo, 'cylinder')
            radius = ET.SubElement(cylinder, 'radius')
            radius.text = str(r)
            length = ET.SubElement(cylinder, 'length')
            length.text = str(l)

    # Store model
    tree.write(os.path.join(os.environ['PWD'], model_dir, 'line.sdf'), encoding="us-ascii", xml_declaration=True,
               method="xml")
    return x_g, y_g, z_g


def spawn_line():
    reference_pos = update_line_model()
    args = shlex.split("rosrun gazebo_ros spawn_model -file " + os.environ[
        "GAZEBO_MODEL_PATH"] + "/line_segment/line.sdf -sdf -model line -y 0 -x 0 " + ("-z 0.1" if WORLD == "gate_cone_line_realistic" else ""))
    subprocess.run(args)
    return reference_pos


def remove_line():
    args = shlex.split("rosservice call gazebo/delete_model '{model_name: line}'")
    subprocess.run(args)


def send_reference_and_set_goal(x: float = 0, y: float = 0, z: float = 0):
    args = shlex.split("rostopic pub /waypoint_indicator/current_waypoint std_msgs/Float32MultiArray '{data:[" + str(x) +", " + str(y) +", "+str(z)+"]}'")
    p = subprocess.Popen(args)
    time.sleep(1)
    p.terminate()
    margin = 0.5
    args = shlex.split("rosparam set /world/goal '{x: {min: " + str(x - margin) + ", max: " + str(x + margin) + "}, y: {min: "+ str(y - margin)+ ", max: "+ str(y + margin) + "}, z: {min: " + str(z - margin) + ", max: "+str(z + margin)+"}}'")
    subprocess.run(args)
    args = shlex.split("rosparam set /starting_height '"+str(z)+"'")
    subprocess.run(args)


def set_random_gate_location():
    set_model_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    model_state = ModelState()
    model_state.pose = Pose()
    model_state.model_name = 'gate'
    model_state.pose.position.x = np.random.uniform(1, 4)
    model_state.pose.position.y = np.random.uniform(-model_state.pose.position.x, model_state.pose.position.x)
    yaw = np.sign(model_state.pose.position.y) * np.random.uniform(0, 30) * np.pi / 180
    model_state.pose.orientation.w = np.cos(yaw * 0.5)
    model_state.pose.orientation.z = np.sin(yaw * 0.5)
    set_model_state_service.wait_for_service()
    set_model_state_service(model_state)
    x = model_state.pose.position.x
    y = model_state.pose.position.y
    z = np.random.uniform(1.2, 1.7)
    send_reference_and_set_goal(x, y, z)
    return x, y, z


def set_random_cone_location():
    set_model_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    model_state = ModelState()
    model_state.pose = Pose()
    model_state.model_name = 'cone'
    model_state.pose.position.x = np.random.uniform(1.5, 4)
    model_state.pose.position.y = np.random.uniform(-model_state.pose.position.x/2, model_state.pose.position.x/2)
    model_state.pose.position.z = 0.1 if WORLD == 'gate_cone_line_realistic' else 0
    set_model_state_service.wait_for_service()
    set_model_state_service(model_state)
    z = np.random.uniform(0.7, 1.7)
    x = model_state.pose.position.x
    y = model_state.pose.position.y
    send_reference_and_set_goal(x, y, z)
    return x, y, z


def spawn_flying_zone():
    subprocess.run(shlex.split("rosrun gazebo_ros spawn_model -database flyzone_wall -sdf -model flyzone_wall -x 12 -y 2 -Y -1.57"))


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

    # don't save is item not in view
    if np.sum(mask) < (1000 if TARGET == 'gate' else 20):
        return

    # store mask
    hdf5_data["mask"].append(mask)

    # store previous observation
    hdf5_data["observation"].append(deepcopy(experience.observation))

    # store velocity
    action = experience.action.value
    json_data["velocities"].append([action[0], action[1], action[2], action[5]])

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

    environment._clear_experience_values()
    environment._pause_period = 1 / 100.
    while environment.observation is None:
        set_model_state_service.call(model_state)
        environment._run_shortly()

    overtake_p.terminate()
    speed_p.terminate()


if __name__ == '__main__':

    robot_dict = {
        'gate': "drone_sim_forward_cam",
        'cone': "drone_sim",
        'line': "drone_sim_down_cam"
    }

    environment_config_dict = {
        "output_path": f"{WORLD}/{TARGET}",
        "factory_key": "ROS",
        "max_number_of_steps": 100,
        "ros_config": {
            "observation": "camera",
            "info": ["position"],
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
                "control_mapping_config": "mathias_controller",
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
    environment = RosEnvironment(
        config=config
    )
    time.sleep(5)
    run_index = 0
    while run_index < NUMBER:
        # create output json and hdf5 file
        json_data = {
            "velocities": [],
            "global_target_location": [],
            "relative_target_location": [],
            "global_drone_pose": []}
        hdf5_data = {"mask": [],
                     "observation": []}

        # collect info in corresponding objects
        if TARGET == 'cone':
            reference_pos = set_random_cone_location()
        elif TARGET == 'gate':
            reference_pos = set_random_gate_location()
        elif TARGET == 'line':
            reference_pos = spawn_line()
        else:
            raise NotImplementedError

        if WORLD == 'gate_cone_line_realistic':
            # decide at which step to randomly stop
            average_lengths = {'cone': 36, 'line': 32, 'gate': 47}
            stop_index = np.random.uniform(0, average_lengths[TARGET])
            print(f'stopping at {stop_index}')
        else:
            stop_index = -1

        step_index = 0
        experience, _ = environment.reset()
        while experience.done == TerminationType.NotDone:
            experience, observation = environment.step()
            if WORLD == 'gate_cone_line':
                save(reference_pos, experience, json_data, hdf5_data)
            step_index += 1
            if step_index >= stop_index and stop_index != -1:
                break

        if WORLD == 'gate_cone_line_realistic':
            # delete flying zone
            subprocess.run(shlex.split("rosservice call gazebo/delete_model '{model_name: flyzone_wall}'"))
            # freeze drone (or make static) run shortly till till new observation
            run_shortly_while_freezing_drone()
            # extract mask from observation
            mask = deepcopy(environment.observation)
            mask = np.mean(mask, axis=2)
            mask[mask == 1] = 0
            mask[mask != 0] = 1
            save(reference_pos, experience, json_data, hdf5_data, mask)
            # put flying zone back
            spawn_flying_zone()

        if TARGET == 'line':
            remove_line()

        # dump data if run is successfully and at least 3 experiences are saved with something in view.
        if WORLD == 'gate_cone_line_realistic' or \
                (experience.done == TerminationType.Success and len(hdf5_data['mask']) > 5):
            run_index = dump(json_data, hdf5_data, output_dir)
            print(f'{get_date_time_tag()}: stored {run_index} / {NUMBER}')
    print(f'finished {environment.remove()}')
