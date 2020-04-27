#!/usr/bin/python3.7

""" Reason

"""
import os
import time

import rospy
import torch
import yaml
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

from src.ai.model import Model, ModelConfig
from src.core.logger import get_logger, cprint, MessageType
from src.sim.common.actors import Actor, ActorConfig, DnnActor
from src.core.data_types import Action
from src.sim.ros.src.utils import adapt_twist_to_action, process_image, get_output_path, adapt_action_to_twist
from src.core.utils import camelcase_to_snake_format, get_filename_without_extension


class DnnActorRos(DnnActor):

    def __init__(self):
        rospy.init_node('dnn_actor_ros')
        start_time = time.time()
        max_duration = 60
        while not rospy.has_param('/output_path') and time.time() < start_time + max_duration:
            time.sleep(0.1)
        self._specs = rospy.get_param('/actor/dnn_actor/specs')
        super().__init__(
            config=ActorConfig(
                name='dnn_actor',
                specs=self._specs
            )
        )
        self._output_path = get_output_path()
        self._logger = get_logger(get_filename_without_extension(__file__), self._output_path)
        cprint(f'&&&&&&&&&&&&&&&&&& \n {self._specs} \n &&&&&&&&&&&&&&&&&', self._logger)
        with open(os.path.join(self._output_path, 'dnn_actor_specs.yml'), 'w') as f:
            yaml.dump(self._specs, f)
        self._rate_fps = self._specs['rate_fps'] if 'rate_fps' in self._specs.keys() else 20
        self._rate = rospy.Rate(self._rate_fps)
        config_dict = self._specs['model_config']
        config_dict['output_path'] = self._output_path
        cprint(f'loading model...', self._logger)
        self._model = Model(config=ModelConfig().create(config_dict=config_dict))
        self._input_sizes = self._model.get_input_sizes()

        self._publisher = rospy.Publisher(self._specs['command_topic'], Twist, queue_size=10)
        self._subscribe()
        cprint(f'ready', self._logger)

    def _subscribe(self):
        for sensor in ['forward_camera']:  # future camera's could be added according to actor's specs
            if rospy.has_param(f'/robot/{sensor}_topic'):
                sensor_topic = rospy.get_param(f'/robot/{sensor}_topic')
                sensor_type = rospy.get_param(f'/robot/{sensor}_type')
                sensor_callback = f'_process_{camelcase_to_snake_format(sensor_type)}'
                if sensor_callback not in self.__dir__():
                    cprint(f'Could not find sensor_callback {sensor_callback}', self._logger, MessageType.error)
                # sensor_stats = rospy.get_param(f'{sensor}_stats') if rospy.has_param(f'{sensor}_stats') else {}
                print(self._input_sizes)
                sensor_stats = {
                    'height': self._input_sizes[0][1],
                    'width': self._input_sizes[0][2],
                    'depth': self._input_sizes[0][0]}
                rospy.Subscriber(name=sensor_topic,
                                 data_class=eval(sensor_type),
                                 callback=eval(f'self.{sensor_callback}'),
                                 callback_args=(sensor_topic, sensor_stats))

    def _process_image(self, msg: Image, args: tuple) -> None:
        # cprint(f'image received {msg.data}', self._logger)
        sensor_topic, sensor_stats = args
        processed_image = process_image(msg, sensor_stats=sensor_stats)
        processed_image = torch.Tensor(processed_image).permute(2, 0, 1).unsqueeze(0)
        assert processed_image.size()[0] == 1 and processed_image.size()[1] == 3
        output = self._model.forward([processed_image])[0].detach().cpu().numpy()
        cprint(f'output predicted {output}', self._logger, msg_type=MessageType.debug)
        action = Action(
            actor_name='dnn_actor',
            value=output  # assuming control is first output
        )
        self._publisher.publish(adapt_action_to_twist(action))

    def run(self):
        cprint(f'started with rate {self._rate_fps}', self._logger)
        while not rospy.is_shutdown():
            self._rate.sleep()


if __name__ == "__main__":
    dnn_actor = DnnActorRos()
    dnn_actor.run()
