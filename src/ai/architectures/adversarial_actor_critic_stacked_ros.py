#!/bin/python3.8
from typing import Iterator, Union, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

from src.ai.architectures.bc_actor_critic_stochastic_continuous import Net as BaseNet
from src.ai.base_net import ArchitectureConfig
from src.ai.utils import mlp_creator, get_slow_hunt, clip_action_according_to_playfield_size_flipped
from src.core.data_types import Action
from src.core.logger import get_logger, cprint, MessageType
from src.core.utils import get_filename_without_extension
from src.sim.ros.src.utils import calculate_bounding_box

"""
Adversarial agents for gazebo world tracking_y_axis
Observation space contains of 9 dimensional array with
[tracking_x, tracking_y, tracking_z, fleeing_x, fleeing_y, fleeing_z, tracking_roll, tracking_pitch, tracking_yaw]
as specifed by the ROS CombinedGlobalPoses message. 
Observation space is shared by standard (tracking: 0) and adversarial (fleeing: 1) agent.
Action space contains [tracking_linear_x, tracking_linear_y, tracking_linear_z,
                        fleeing_linear_x, fleeing_linear_y, fleeing_linear_z,
                        tracking_angular_z, fleeing_angular_z] 
as specified by src/sim/ros/src/utils/adapt_action_to_twist according to actor_name 'tracking_fleeing'.
The actor (policies) should only predict sideways (tracking_linear_y and fleeing_linear_y) for this architecture.
Game specified action clipping done when agent reaches boundaries of game should be hard coded in this architecture.
"""
EPSILON = 1e-6


#  - clip actions when getting out of play field


class Net(BaseNet):

    def __init__(self, config: ArchitectureConfig, quiet: bool = False):
        super().__init__(config=config, quiet=True)
        self._playfield_size = (1, 1, 0)
        self.input_size = (4,)
        self.output_size = (8,)
        self.action_min = -1
        self.action_max = 1
        self.starting_height = -1
        self.previous_input = torch.Tensor([200, 200, 20, 20])

        self._actor = mlp_creator(sizes=[self.input_size[0], 10, 2],  # for now actors can only fly sideways
                                  activation=nn.Tanh(),
                                  output_activation=None)
        log_std = self._config.log_std if self._config.log_std != 'default' else -0.5
        self.log_std = torch.nn.Parameter(torch.ones((1,), dtype=torch.float32) * log_std,
                                          requires_grad=True)

        self._critic = mlp_creator(sizes=[self.input_size[0], 10, 1],
                                   activation=nn.Tanh(),
                                   output_activation=None)

        self._adversarial_actor = mlp_creator(sizes=[self.input_size[0], 10, 2],
                                              activation=nn.Tanh(),
                                              output_activation=None)
        self.adversarial_log_std = torch.nn.Parameter(torch.ones((1,),
                                                                 dtype=torch.float32) * log_std, requires_grad=True)

        self._adversarial_critic = mlp_creator(sizes=[self.input_size[0], 10, 1],
                                               activation=nn.Tanh(),
                                               output_activation=None)

        if not quiet:
            self._logger = get_logger(name=get_filename_without_extension(__file__),
                                      output_path=config.output_path,
                                      quiet=False)
            cprint(f'Started.', self._logger)
            self.initialize_architecture()

    def adjust_height(self, inputs, actions: np.ndarray) -> np.ndarray:
        if self.starting_height == -1:  # Take average between tracking and fleeing height to be kept constant
            self.starting_height = (inputs[2] + inputs[5]) / 2
        else:
            for agent in ['tracking', 'fleeing']:
                height = inputs[2] if agent == 'tracking' else inputs[5]
                if height < (self.starting_height - 0.1):
                    actions[2 if agent == 'tracking' else 5] = +0.5
                elif height > (self.starting_height + 0.1):
                    actions[2 if agent == 'tracking' else 5] = +0.5
                else:
                    actions[2 if agent == 'tracking' else 5] = +0.5
        return actions

    def get_action(self, inputs, train: bool = False, agent_id: int = -1) -> Action:
        positions = np.squeeze(self.process_inputs(inputs))
        try:
            bb = calculate_bounding_box(inputs, orientation=(0, 0, 1))
            inputs = (bb[3][0], bb[3][1], bb[4], bb[5])
            inputs = np.squeeze(self.process_inputs(inputs))
            self.previous_input = inputs
        except TypeError:
            inputs = self.previous_input
        if agent_id == 0:  # tracking agent ==> tracking_linear_y
            output = self.sample(inputs, train=train).clamp(min=self.action_min, max=self.action_max)
            actions = np.stack([output.data.cpu().numpy().squeeze(), 0, 0, 0, 0, 0, 0, 0])
        elif agent_id == 1:  # fleeing agent ==> fleeing_linear_y
            output = self.sample(inputs, train=train, adversarial=True).clamp(min=self.action_min,
                                                                              max=self.action_max)
            actions = np.stack([0, 0, 0, output.data.cpu().numpy().squeeze(), 0, 0, 0, 0])
        else:
            output = self.sample(inputs, train=train, adversarial=False).clamp(min=self.action_min, max=self.action_max)
            adversarial_output = self.sample(inputs, train=train, adversarial=True).clamp(min=self.action_min,
                                                                                          max=self.action_max)
            actions = np.stack([*output.data.cpu().numpy().squeeze(), 0,
                                *adversarial_output.data.cpu().numpy().squeeze(), 0,
                                0, 0], axis=-1)
            # actions = np.stack([0, -1, 0, 0, -1, 0, 0, 0], axis=-1)

        # actions = self.adjust_height(positions, actions)  Not necessary, controller keeps altitude fixed
        actions = clip_action_according_to_playfield_size_flipped(positions.detach().numpy().squeeze(),
                                                                  actions, self._playfield_size)
        return Action(actor_name="tracking_fleeing_agent",  # assume output [1, 8] so no batch!
                      value=actions)

    def _policy_distribution(self, inputs: torch.Tensor,
                             train: bool = True, adversarial: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        self.set_mode(train)
        inputs = self.process_inputs(inputs=inputs)
        means = self._actor(inputs) if not adversarial else self._adversarial_actor(inputs)
        return means, torch.exp(self.log_std if not adversarial else self.adversarial_log_std)

    def sample(self, inputs: torch.Tensor, train: bool = False, adversarial: bool = False) -> torch.Tensor:
        mean, std = self._policy_distribution(inputs, train=train, adversarial=adversarial)
        return (mean + torch.randn_like(mean) * std).detach()

    def get_adversarial_policy_entropy(self, inputs: torch.Tensor, train: bool = True) -> torch.Tensor:
        mean, std = self._policy_distribution(inputs=inputs, train=train, adversarial=True)
        return Normal(mean, std).entropy().sum(dim=1)

    def policy_log_probabilities(self, inputs, actions, train: bool = True, adversarial: bool = False) -> torch.Tensor:
        actions = self.process_inputs(inputs=[a[:2] if not adversarial else a[3:5] for a in actions])
        try:
            mean, std = self._policy_distribution(inputs, train, adversarial)
            log_probabilities = -(0.5 * ((actions - mean) / (std + EPSILON)).pow(2).sum(-1) +
                                  0.5 * np.log(2.0 * np.pi) * actions.shape[-1]
                                  + (self.log_std.sum(-1) if not adversarial else self.adversarial_log_std.sum(-1)))
            return log_probabilities
        except Exception as e:
            raise ValueError(f"Numerical error: {e}")

    def adversarial_policy_log_probabilities(self, inputs, actions, train: bool = True) -> torch.Tensor:
        return self.policy_log_probabilities(inputs, actions, train, adversarial=True)

    def critic(self, inputs, train: bool = False) -> torch.Tensor:
        if len(inputs[0]) == 9:
            for index in range(len(inputs)):
                try:
                    bb = calculate_bounding_box(np.asarray(inputs[index]), orientation=(0, 0, 1))
                    inputs[index] = torch.Tensor([bb[3][0], bb[3][1], bb[4], bb[5]])
                except TypeError:
                    if index == 0:
                        inputs[index] = torch.Tensor([200, 200, 20, 20])
                    else:
                        inputs[index] = inputs[index - 1]
        self._critic.train()
        inputs = np.squeeze(self.process_inputs(inputs=inputs))
        return self._critic(inputs)

    def adversarial_critic(self, inputs, train: bool = False) -> torch.Tensor:
        self._adversarial_critic.train(train)
        inputs = self.process_inputs(inputs=inputs)
        return self._adversarial_critic(inputs)

    def get_adversarial_actor_parameters(self) -> Iterator:
        for p in [self.adversarial_log_std, *self._adversarial_actor.parameters()]:
            yield p

    def get_adversarial_critic_parameters(self) -> Iterator:
        return self._adversarial_critic.parameters()
