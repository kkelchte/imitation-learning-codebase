import os

from src.sim.ros.src.process_wrappers import RosWrapper

DS_TASK = 'waypoints'  # 'velocities'  # waypoints
JOY = True # False

if __name__ == '__main__':
    if DS_TASK == 'waypoints':
        control_mapping_config = 'mathias_controller_joystick' if JOY else 'mathias_controller_keyboard'
    else:
        control_mapping_config = 'joystick_python' if JOY else 'keyboard_python'

    config = {
        'output_path': "real-bebop-fgbg",
        'robot_name': 'bebop_real',
        'gazebo': False,
        'fsm': True,
        'fsm_mode': 'TakeOverRun',
        'control_mapping': True,
        'control_mapping_config': control_mapping_config,
        'april_tag_detector': False,
        'altitude_control': False,
        'robot_display': True,
        'mathias_controller_with_KF': DS_TASK == "waypoints",
        'keyboard': not JOY,
        'joystick': JOY,
        'mathias_controller_config_file_path_with_extension':
            f'{os.environ["CODEDIR"]}/src/sim/ros/config/actor/mathias_controller_with_KF_real_bebop.yml',
    }
    
    # spinoff roslaunch
    ros_process = RosWrapper(launch_file='load_ros.launch',
                             config=config,
                             visible=True)
