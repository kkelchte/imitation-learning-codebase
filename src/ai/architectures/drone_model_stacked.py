#!/bin/python3.8
from PIL import Image
from src.ai.architectures.bc_actor_critic_stochastic_continuous import Net as BaseNet
from src.ai.base_net import ArchitectureConfig
from src.ai.utils import mlp_creator, find_person
from src.core.data_types import Action
from src.core.logger import get_logger, cprint, MessageType
from src.core.utils import get_filename_without_extension
from src.ai.yolov3.yolov3_pytorch.utils import *
from src.ai.yolov3.yolov3_pytorch.yolov3_tiny import *


class Net(BaseNet):

    def __init__(self, config: ArchitectureConfig, quiet: bool = False):
        super().__init__(config=config, quiet=True)
        self.input_size = (2,)
        self.output_size = (8,)
        self.action_max = 0.5
        self.starting_height = -1
        self.previous_input = torch.Tensor([0, 0])
        self.sz = 416

        self._actor = mlp_creator(sizes=[self.input_size[0], 8, 2],  # for now actors can only fly sideways
                                  layer_bias=True,
                                  activation=nn.Tanh(),
                                  output_activation=nn.Tanh())
        log_std = self._config.log_std if self._config.log_std != 'default' else -0.5
        self.log_std = torch.nn.Parameter(torch.ones((1,), dtype=torch.float32) * log_std,
                                          requires_grad=True)

        self._critic = mlp_creator(sizes=[self.input_size[0], 8, 1],
                                   layer_bias=True,
                                   activation=nn.Tanh(),
                                   output_activation=nn.Tanh())

        self._adversarial_actor = mlp_creator(sizes=[self.input_size[0], 8, 2],
                                              layer_bias=True,
                                              activation=nn.Tanh(),
                                              output_activation=nn.Tanh())
        self.adversarial_log_std = torch.nn.Parameter(torch.ones((1,),
                                                                 dtype=torch.float32) * log_std, requires_grad=True)

        self._adversarial_critic = mlp_creator(sizes=[self.input_size[0], 8, 1],
                                               layer_bias=True,
                                               activation=nn.Tanh(),
                                               output_activation=nn.Tanh())

        self.yolov3_tiny = Yolov3Tiny(num_classes=80)
        self.yolov3_tiny.load_state_dict(torch.load('src/ai/yolov3/yolov3_files/yolov3_tiny_coco_01.h5'))

        if not quiet:
            self._logger = get_logger(name=get_filename_without_extension(__file__),
                                      output_path=config.output_path,
                                      quiet=False)
            cprint(f'Started.', self._logger)
            self.initialize_architecture()

    def get_action(self, input_img: Image, train: bool = False, agent_id: int = -1) -> Action:
        try:
            img = image2torch(input_img.resize((self.sz, self.sz)))
            boxes = self.yolov3_tiny.predict_img(img, conf_thresh=0.7)[0]
            inputs = find_person(boxes, self.previous_input)
            inputs = np.squeeze(self.process_inputs(inputs))
            self.previous_input = inputs
        except TypeError:
            inputs = self.previous_input

        output = self.action_max * self.sample(inputs, train=False)
        actions = np.stack([*output.data.cpu().numpy().squeeze(), 0, 0, 0], axis=-1)

        return Action(actor_name="tracking_fleeing_agent",  # assume output [1, 8] so no batch!
                      value=actions)
