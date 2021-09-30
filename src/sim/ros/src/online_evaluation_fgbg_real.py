import os
import sys
import time

import rospy
from geometry_msgs.msg import Point, PointStamped
from sensor_msgs.msg import Image
import fgbg
import torch

from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.src.utils import process_image

TARGET = 'cone'
DS_TASK = 'waypoints'  # 'velocities'  # waypoints
CHECKPOINT = f'{os.environ["HOME"]}/code/contrastive-learning/data/down_stream/{DS_TASK}/{TARGET}/best/checkpoint_model.ckpt'

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

    if DS_TASK == "velocities":
        model = fgbg.DownstreamNet(
            output_size=(4,),
            batch_norm=True,
        )
    else:
        model = fgbg.DownstreamNet(
            output_size=(3,),
            batch_norm=True
        )

    ckpt = torch.load(CHECKPOINT, map_location=torch.device("cpu"))
    model.load_state_dict(ckpt["state_dict"])
    model.global_step = ckpt["global_step"]
    model.eval()
    # TODO ADD GPU OPTION
    

    def evaluate_image(msg: Image):
        image = process_image(msg, {'height': 200, 'width': 200, 'depth': 3})
        tensor = torch.as_tensor(image).permute(2, 0, 1).unsqueeze(0)
        prediction = model(tensor).squeeze().cpu().detach().numpy()
        point = Point(x=prediction[0].item(), y=prediction[1].item(), z=prediction[2].item())
        reference_publisher.publish(PointStamped(point=point))

    rospy.init_node('online_evaluation_fgbg_real')
    reference_publisher = rospy.Publisher(name='/reference_pose',
                                          data_class=PointStamped,
                                          queue_size=10)
    rospy.Subscriber(name='/bebop/image_raw',
                     data_class=Image,
                     callback=evaluate_image)

    rate = rospy.Rate(60)
    while not rospy.is_shutdown():
        rate.sleep()
        


