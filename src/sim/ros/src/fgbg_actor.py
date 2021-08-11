import os
from sys import argv
import shutil
import time
import unittest
from copy import deepcopy
from typing import Union

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

class Actor:

    def __init__(self, task: str, ckpt: str = None):
        rospy.init_node('fgbg_actor')
        self._task = task
        # Load model
        if task == 'pretrain':
            self.model = fgbg.DeepSupervisionNet(batch_norm=True)
        else:
            self.model = fgbg.DownstreamNet(output_size=(4,) if self._task == "velocities" else (3,), batch_norm=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if ckpt is not None:
            ckpt = torch.load(
                ckpt + "/checkpoint_model.ckpt", map_location=torch.device("cpu")
            )
            self.model.load_state_dict(ckpt["state_dict"])
            print(f"Loaded encoder from {ckpt} on {self.device}.")
        
        self.model.to(self.device)
        # wait till ROS started properly
        stime = time.time()
        max_duration = 60
        while not rospy.has_param('/output_path') and time.time() < stime + max_duration:
            time.sleep(0.01)
        rospy.Subscriber(name=rospy.get_param(f'/robot/camera_sensor/topic'),
                         data_class=eval(rospy.get_param(f'/robot/camera_sensor/type')),
                         callback=self._camera_callback)
        self._reset_publisher = rospy.Publisher("/fsm/reset", Empty, queue_size=10)
        if self._task == 'waypoints': 
            self._prediction_publisher = rospy.Publisher(
                "/reference_pose", PointStamped, queue_size=10
            )
        else:
            self._prediction_publisher = rospy.Publisher(
                f'/ros_python_interface/cmd_vel',
                Twist,
                queue_size=10,
            )

        self._mask_publisher = rospy.Publisher('/mask', Image, queue_size=10)

    def reset(self):
        self._reset_publisher(Empty())

    def _camera_callback(self, msg: Image):
        image = process_image(msg, {'height': 200, 'width': 200, 'depth': 3})
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        
        prediction = self.model(image_tensor).detach().cpu().numpy().squeeze()
        if self._task != 'pretrain':
            self._publish(prediction)       
            mask = self.model.encoder(image_tensor).detach().cpu().numpy().squeeze()
        else:
            mask = prediction
        mask = np.stack([mask] * 3, axis=-1)
        self._publish_mask(mask)

    def _publish_mask(self, mask):
        msg = Image()
        msg.data = list((mask * 255).astype(np.uint8).flatten())
        msg.height = 200
        msg.width = 200
        msg.encoding = 'rgb8'
        self._mask_publisher.publish(msg)

    def _publish(self, prediction):
        if self._task == 'waypoints':
            msg = PointStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "agent"
            msg.point = Point(prediction[0], prediction[1], prediction[2])
        else:
            msg = adapt_action_to_twist(Action(name='agent', value=prediction))        
        self._prediction_publisher.publish(msg)

    def run(self):
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            rate.sleep()


if __name__ == '__main__':
    task = 'pretrain'
    target = 'line'
    config = {
        'cone': 'deep_supervision_blur',
        'gate': 'deep_supervision_blur',
        'line': 'deep_supervision_blur'
    }
    lrs = {
        'cone': 1e-05,
        'gate': 1e-05,
        'line': 0.01
    }
    # ckpt = os.path.join(os.environ['HOME'], 'code/contrastive-learning/data/best_down_stream', task, target)
    ckpt = os.path.join(os.environ['HOME'], 'mount/esat/code/contrastive-learning/data/dtd_augmented', task, config[target], target, str(lrs[target]))
    assert os.path.isdir(ckpt)
    actor = Actor(task, ckpt)
    actor.run()