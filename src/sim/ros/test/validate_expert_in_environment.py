import os
import shutil
import unittest

import numpy as np
import rospy
import yaml

from src.core.utils import get_filename_without_extension
from src.sim.common.data_types import TerminalType
from src.sim.common.environment import EnvironmentConfig
from src.sim.ros.src.ros_environment import RosEnvironment
from src.sim.ros.catkin_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState

#  Define the environment you want to test:
world_name = 'cube_world'

output_dir = f'test_dir/{get_filename_without_extension(__file__)}'


def setup() -> EnvironmentConfig:
    config_file = 'test_ros_environment'
    os.makedirs(output_dir, exist_ok=True)
    with open(f'src/sim/ros/test/config/{config_file}.yml', 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config_dict['output_path'] = output_dir
    config_dict['ros_config']['ros_launch_config']['world_name'] = world_name
    return EnvironmentConfig().create(
        config_dict=config_dict
    )


def tear_down(environment: RosEnvironment) -> None:
    environment.remove()
    shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == '__main__':
    config = setup()
    environment = RosEnvironment(
        config=config
    )
    state = environment.reset()

    # wait delay evaluation time
    while state.terminal == TerminalType.Unknown:
        state = environment.step()
    print(f'finished startup')
    waypoints = rospy.get_param('/world/waypoints')

    for waypoint_index, waypoint in enumerate(waypoints[:-1]):
        print(f'started with waypoint: {waypoint}')
        while state.sensor_data['current_waypoint'].tolist() == waypoint:
            state = environment.step()

    print(f'ending with waypoint {waypoints[-1]}')
    while not environment.fsm_state == FsmState.Terminated:
        state = environment.step()
    print(f'terminal type: {state.terminal.name}')
    tear_down(environment)
