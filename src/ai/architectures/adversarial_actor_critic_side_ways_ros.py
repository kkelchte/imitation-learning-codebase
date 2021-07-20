#!/bin/python3.8
from typing import Iterator, Union, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

from src.ai.architectures.bc_actor_critic_stochastic_continuous import Net as BaseNet
from src.ai.base_net import ArchitectureConfig
from src.ai.utils import mlp_creator, get_waypoint, clip_action_according_to_playfield_size_flipped, get_slow_run_ros, \
    get_slow_hunt_ros, get_rand_run_ros
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
        self._playfield_size = (0, 1, 0)
        self.input_size = (1,)
        self.output_size = (8,)
        self.action_min = -0.5
        self.action_max = 0.5
        self.starting_height = -1
        self.previous_input = 0

        self.waypoint = get_waypoint(self._playfield_size)

        self._actor = mlp_creator(sizes=[self.input_size[0], 4, 1],  # for now actors can only fly sideways
                                  layer_bias=False,
                                  activation=nn.Tanh(),
                                  output_activation=None)
        log_std = self._config.log_std if self._config.log_std != 'default' else -0.5
        self.log_std = torch.nn.Parameter(torch.ones((1,), dtype=torch.float32) * log_std,
                                          requires_grad=True)

        self._critic = mlp_creator(sizes=[self.input_size[0], 4, 1],
                                   layer_bias=False,
                                   activation=nn.Tanh(),
                                   output_activation=None)

        self._adversarial_actor = mlp_creator(sizes=[self.input_size[0], 4, 1],
                                              layer_bias=False,
                                              activation=nn.Tanh(),
                                              output_activation=None)
        self.adversarial_log_std = torch.nn.Parameter(torch.ones((1,),
                                                                 dtype=torch.float32) * log_std, requires_grad=True)

        self._adversarial_critic = mlp_creator(sizes=[self.input_size[0], 4, 1],
                                               layer_bias=False,
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
            bb = calculate_bounding_box(inputs)
            inputs_bb = (bb[3][0], bb[3][1], bb[4], bb[5])
            inputs_ = (bb[3][0] - 200)/40
            inputs = torch.Tensor([inputs_])
            self.previous_input = inputs_
        except TypeError:
            inputs_bb = (200, 200, 40, 40)
            inputs = torch.Tensor([self.previous_input])
        if agent_id == 0:  # tracking agent ==> tracking_linear_y
            output = self.sample(inputs, train=train).clamp(min=self.action_min, max=self.action_max)
            actions = np.stack([0, output.data.cpu().numpy().squeeze(), 0, 0, 0, 0, 0, 0])
        elif agent_id == 1:  # fleeing agent ==> fleeing_linear_y
            output = self.sample(inputs, train=train, adversarial=True).clamp(min=self.action_min,
                                                                              max=self.action_max)
            actions = np.stack([0, 0, 0, 0, output.data.cpu().numpy().squeeze(), 0, 0, 0])
        else:
            output = self.sample(inputs, train=train, adversarial=False).clamp(min=self.action_min, max=self.action_max)
            adversarial_output = self.sample(inputs, train=train, adversarial=True).clamp(min=self.action_min,
                                                                                          max=self.action_max)

            # rand_run = get_rand_run_ros(self.waypoint, np.asarray(positions[3:6]).squeeze(), self._playfield_size)
            # run_action = np.squeeze(rand_run[1])
            # self.waypoint = np.squeeze(rand_run[0])
            # hunt_action = np.squeeze(get_slow_hunt_ros(np.asarray(positions), self._playfield_size))
            # run_action = np.squeeze(get_slow_run_ros(inputs_bb, self._playfield_size))


            actions = np.stack([0, output.data.cpu().numpy().squeeze().item(), 0,
                                0, adversarial_output.data.cpu().numpy().squeeze().item(), 0,
                                0, 0], axis=-1)

            # actions = np.stack([0, output.data.cpu().numpy().squeeze().item(), 0, *run_action, 0, 0], axis=-1)

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
        actions = self.process_inputs(inputs=[a[1] if not adversarial else a[4] for a in actions])

        try:
            mean, std = self._policy_distribution(inputs, train, adversarial)
            actions = actions.squeeze()
            mean = mean.transpose(0, 1).squeeze()
            log_probabilities = -(0.5 * ((actions - mean) / (std + EPSILON)).pow(2) + 0.5 * np.log(2.0 * np.pi)
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
                    bb = calculate_bounding_box(np.asarray(inputs[index]))
                    if index == 0:
                        inputs[index] = torch.Tensor([(bb[3][0]-200)/40])
                    else:
                        inputs[index] = torch.Tensor([(bb[3][0]-200)/40])
                except (TypeError, IndexError):
                    if index == 0:
                        inputs[index] = torch.Tensor([0])
                    else:
                        inputs[index] = inputs[index - 1]
        self._critic.train()
        inputs = self.process_inputs(inputs=inputs)
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
