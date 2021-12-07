#!/usr/bin/python3.8
import os
from sys import argv
import shutil
import time
import unittest
from copy import deepcopy
from typing import Union

import torch
import matplotlib.pyplot as plt
from std_msgs.msg import Empty
import numpy as np
import rospy
from PIL import Image as PILImage 
from PIL import ImageEnhance as PILImageEnhance
from geometry_msgs.msg import Twist, PointStamped, Point
from std_msgs.msg import String, Float32MultiArray, Empty
from sensor_msgs.msg import Image
from scipy.interpolate import interp1d
import fgbg

from src.core.data_types import Action
from src.sim.ros.src.utils import process_image, adapt_action_to_twist

class Actor:

    def __init__(self, task: str, ckpt: str = None, batch_norm: bool = False, enhance_brightness: bool = False):
        rospy.init_node('fgbg_actor')
        self._task = task
        self._enhance_brightness = enhance_brightness
        # Load model
        if task == 'pretrain':
            self.model = fgbg.DeepSupervisionNet(batch_norm=batch_norm)
        else:
            self.model = fgbg.DownstreamNet(output_size=(4,) if self._task == "velocities" else (3,), batch_norm=batch_norm)
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
        self._last_image = None
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

        self._mask_publisher = rospy.Publisher('/mask', Image, queue_size=1)

    def reset(self):
        self._reset_publisher(Empty())
        self._last_image = None

    def _camera_callback(self, msg: Image):
        self._last_image  = msg

    def _publish_mask(self, mask):
        msg = Image()
        # mask -= mask.min()
        # mask /= mask.max()
        mask = (mask * 255.).flatten().astype(np.uint8)
        msg.data = [m for m in mask]
        msg.height = 200
        msg.width = 200
        msg.encoding = 'mono8'
        self._mask_publisher.publish(msg)

    def _publish_prediction(self, prediction):
        if self._task == 'waypoints':
            msg = PointStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "agent"
            msg.point = Point(prediction[0], prediction[1], prediction[2])
        else:
            msg = Twist()
            msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z = tuple(prediction)     
        self._prediction_publisher.publish(msg)

    def _update(self):
        if self._last_image is not None:
            image = deepcopy(self._last_image)
            self._last_image = None
            image = process_image(image, {'height': 200, 'width': 200, 'depth': 3})
            if self._enhance_brightness:
                original_mean = image.mean()*255
                enhance_factor = 140 / original_mean
                image = PILImage.fromarray(np.uint8(image * 255))
                enhancer = PILImageEnhance.Brightness(image)
                im_output = np.asarray(enhancer.enhance(enhance_factor))
                image_tensor = torch.from_numpy(im_output/255).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
            else:
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
            prediction = self.model(image_tensor).detach().cpu().numpy().squeeze()
            if self._task != 'pretrain':
                self._publish_prediction(prediction)       
                mask = self.model.encoder(image_tensor).detach().cpu().numpy().squeeze()
            else:
                mask = prediction
            self._publish_mask(mask)

    def run(self):
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            self._update()
            rate.sleep()


if __name__ == '__main__':
    # REDLINE 0.0
    # task = 'waypoints'
    # target = 'red_line'
    # ckpt = os.path.join(os.environ['HOME'], 'code/contrastive-learning/data/down_stream', task, target, '1e-05')
    # batch_norm = False
    # enhance_brightness = True

    # BLUE LINE WAYPOINTS
    task = 'waypoints'
    target = 'line'
    # ckpt = os.path.join(os.environ['HOME'], 'code/imitation-learning-codebase/experimental_data/line/baseline')
    # ckpt = os.path.join(os.environ['HOME'], 'code/imitation-learning-codebase/experimental_data/line/deep_supervision')
    ckpt = os.path.join(os.environ['HOME'], 'code/imitation-learning-codebase/experimental_data/line/deep_supervision_triplet')
    batch_norm = False
    enhance_brightness = False

    # BLUE LINE VELOCITIES
    # task = 'velocities'
    # target = 'line'
    # ckpt = os.path.join(os.environ['HOME'], 'code/contrastive-learning/data/down_stream', task, target, '1e-05')
    # batch_norm = False
    # enhance_brightness = True

    # GATE PRETRAIN
    # task = 'waypoints'
    # target = 'gate'
    # ckpt = os.path.join(os.environ['HOME'], 'code/contrastive-learning/data/gate/deep_supervision_comb_blur_brightness_hue_bn/waypoints')
    # batch_norm = True
    # enhance_brightness = True

    assert os.path.isdir(ckpt)
    actor = Actor(task, ckpt, batch_norm, enhance_brightness)
    actor.run()