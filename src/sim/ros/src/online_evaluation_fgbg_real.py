import os

from src.sim.ros.src.process_wrappers import RosWrapper

DS_TASK = 'waypoints'  # 'velocities'  # waypoints

if __name__ == '__main__':
    config = {
        'output_path': "real-bebop-fgbg",
        'robot_name': 'bebop_real',
        'gazebo': False,
        'fsm': True,
        'fsm_mode': 'TakeOverRun',
        'control_mapping': True,
        'control_mapping_config': 'mathias_controller_keyboard' if DS_TASK == "waypoints" else "keyboard_python",
        'april_tag_detector': False,
        'altitude_control': False,
        'robot_display': False,
        'mathias_controller_with_KF': DS_TASK == "waypoints",
        'keyboard': True,
        'mathias_controller_config_file_path_with_extension':
            f'{os.environ["CODEDIR"]}/src/sim/ros/config/actor/mathias_controller_with_KF_real_bebop.yml',
    }
    
    # spinoff roslaunch
    ros_process = RosWrapper(launch_file='load_ros.launch',
                             config=config,
                             visible=True)
