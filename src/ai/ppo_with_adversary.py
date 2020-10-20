#!/usr/bin/python3.8
import torch
from torch import nn
import numpy as np

from src.ai.base_net import BaseNet
from src.ai.trainer import TrainerConfig
from src.ai.utils import get_reward_to_go, get_checksum_network_parameters
from src.ai.ppo import ProximatePolicyGradient
from src.core.data_types import Distribution, Dataset
from src.core.logger import get_logger, cprint
from src.core.tensorboard_wrapper import TensorboardWrapper
from src.core.utils import get_filename_without_extension

"""Given model, config, data_loader, trains a model and logs relevant training information

If later more complex algorithms for RL are to be implemented, they should inherent from here.
Allows combination of outputs as weighted sum in one big backward pass.
"""


class AdversarialProximatePolicyGradient(ProximatePolicyGradient):

    def __init__(self, config: TrainerConfig, network: BaseNet, quiet=False):
        super().__init__(config, network, quiet=True)
        # PPO base class initializes hyperparameters
        # VPG base class initializes actor & critic optimizers (standard)
        # current implementation assumes equal learning rate between standard and adversary agent
        # add adversarial actor & critic optimizers
        kwargs = {'params': self._net.get_adversarial_actor_parameters(),
                  'lr': self._config.learning_rate if self._config.actor_learning_rate == -1
                  else self._config.actor_learning_rate}
        if self._config.optimizer == 'Adam':
            kwargs['eps'] = 1e-05
        self._adversarial_actor_optimizer = eval(f'torch.optim.{self._config.optimizer}')(**kwargs)

        kwargs = {'params': self._net.get_adversarial_critic_parameters(),
                  'lr': self._config.learning_rate if self._config.critic_learning_rate == -1 else
                  self._config.critic_learning_rate}
        if self._config.optimizer == 'Adam':
            kwargs['eps'] = 1e-05
        self._adversarial_critic_optimizer = eval(f'torch.optim.{self._config.optimizer}')(**kwargs)

        lambda_function = lambda f: 1 - f / self._config.scheduler_config.number_of_epochs
        self._adversarial_actor_scheduler = torch.optim.lr_scheduler.LambdaLR(self._adversarial_actor_optimizer,
                                                                              lr_lambda=lambda_function) \
            if self._config.scheduler_config is not None else None
        self._adversarial_critic_scheduler = torch.optim.lr_scheduler.LambdaLR(self._adversarial_critic_optimizer,
                                                                               lr_lambda=lambda_function) \
            if self._config.scheduler_config is not None else None

        if not quiet:
            self._logger = get_logger(name=get_filename_without_extension(__file__),
                                      output_path=config.output_path,
                                      quiet=True)
            cprint(f'Started.', self._logger)
            cprint(f'actor checksum {get_checksum_network_parameters(self._net.get_actor_parameters())}')
            cprint(f'critic checksum {get_checksum_network_parameters(self._net.get_critic_parameters())}')
            cprint(f'adversarial actor checksum '
                   f'{get_checksum_network_parameters(self._net.get_adversarial_actor_parameters())}')
            cprint(f'adversarial critic checksum '
                   f'{get_checksum_network_parameters(self._net.get_adversarial_critic_parameters())}')

    def _train_adversarial_actor_ppo(self, batch: Dataset, phi_weights: torch.Tensor, writer: TensorboardWrapper = None) \
            -> Distribution:
        original_log_probabilities = self._net.adversarial_policy_log_probabilities(inputs=batch.observations,
                                                                                    actions=batch.actions,
                                                                                    train=False).detach()
        list_batch_loss = []
        list_entropy_loss = []
        for _ in range(self._config.max_actor_training_iterations
                       if self._config.max_actor_training_iterations != -1 else 1000):
            for data in self.data_loader.split_data(np.zeros((0,)),  # provide empty array if all data can be selected
                                                    batch.observations,
                                                    batch.actions,
                                                    original_log_probabilities,
                                                    phi_weights):
                mini_batch_observations, mini_batch_actions, \
                    mini_batch_original_log_probabilities, mini_batch_phi_weights = data

                # normalize advantages (phi_weights)
                mini_batch_phi_weights = (mini_batch_phi_weights - mini_batch_phi_weights.mean()) \
                    / (mini_batch_phi_weights.std() + 1e-8)

                new_log_probabilities = self._net.adversarial_policy_log_probabilities(inputs=mini_batch_observations,
                                                                                       actions=mini_batch_actions,
                                                                                       train=True)
                ratio = torch.exp(new_log_probabilities - mini_batch_original_log_probabilities)
                unclipped_loss = ratio * mini_batch_phi_weights
                clipped_loss = ratio.clamp(1 - self._config.ppo_epsilon, 1 + self._config.ppo_epsilon) \
                    * mini_batch_phi_weights
                surrogate_loss = - torch.min(unclipped_loss, clipped_loss).mean()
                entropy_loss = - self._config.entropy_coefficient * \
                    self._net.get_adversarial_policy_entropy(mini_batch_observations, train=True).mean()

                batch_loss = surrogate_loss + entropy_loss
                kl_approximation = (mini_batch_original_log_probabilities - new_log_probabilities).abs().mean().item()
                if kl_approximation > 1.5 * self._config.kl_target and self._config.use_kl_stop:
                    break
                self._adversarial_actor_optimizer.zero_grad()
                batch_loss.backward()
                if self._config.gradient_clip_norm != -1:
                    nn.utils.clip_grad_norm_(self._net.get_adversarial_actor_parameters(),
                                             self._config.gradient_clip_norm)
                self._adversarial_actor_optimizer.step()
                list_batch_loss.append(batch_loss.detach())
                list_entropy_loss.append(entropy_loss.detach())
        actor_loss_distribution = Distribution(torch.stack(list_batch_loss))
        if writer is not None:
            writer.set_step(self._net.global_step)
            writer.write_distribution(actor_loss_distribution, "adversarial_policy_loss")
            writer.write_distribution(Distribution(torch.stack(list_entropy_loss)), "adversarial_policy_entropy_loss")
            writer.write_scalar(list_batch_loss[-1].item(), 'adversarial_final_policy_loss')
            writer.write_scalar(kl_approximation, 'adversarial_kl_difference')
        return actor_loss_distribution

    def _train_adversarial_critic_clipped(self, batch: Dataset, targets: torch.Tensor, previous_values: torch.Tensor) \
            -> Distribution:
        critic_loss = []
        for value_train_it in range(self._config.max_critic_training_iterations):
            state_indices = np.asarray([index for index in range(len(batch)) if not batch.done[index]])
            for data in self.data_loader.split_data(state_indices,
                                                    batch.observations,
                                                    targets,
                                                    previous_values):
                self._adversarial_critic_optimizer.zero_grad()
                mini_batch_observations, mini_batch_targets, mini_batch_previous_values = data

                batch_values = self._net.adversarial_critic(inputs=mini_batch_observations, train=True).squeeze()
                unclipped_loss = self._criterion(batch_values, mini_batch_targets)
                # absolute clipping
                clipped_values = mini_batch_previous_values + \
                    (batch_values - mini_batch_previous_values).clamp(-self._config.ppo_epsilon,
                                                                      self._config.ppo_epsilon)
                clipped_loss = self._criterion(clipped_values, mini_batch_targets)
                batch_loss = torch.max(unclipped_loss, clipped_loss)
                batch_loss.mean().backward()
                if self._config.gradient_clip_norm != -1:
                    nn.utils.clip_grad_norm_(self._net.get_adversarial_critic_parameters(),
                                             self._config.gradient_clip_norm)
                self._adversarial_critic_optimizer.step()
                critic_loss.append(batch_loss.mean().detach())
        return Distribution(torch.stack(critic_loss))

    def train(self, epoch: int = -1, writer=None) -> str:
        self.put_model_on_device()
        batch = self.data_loader.get_dataset()
        assert len(batch) != 0

        # train standard agent
        values = self._net.critic(inputs=batch.observations, train=False).squeeze().detach()
        phi_weights = self._calculate_phi(batch, values).to(self._device).squeeze(-1).detach()
        critic_targets = get_reward_to_go(batch).to(self._device) if self._config.phi_key != 'gae' else \
            (values + phi_weights).detach()
        critic_loss_distribution = self._train_critic_clipped(batch, critic_targets, values)
        actor_loss_distribution = self._train_actor_ppo(batch, phi_weights, writer)
        if writer is not None:
            writer.write_distribution(critic_loss_distribution, "critic_loss")
            writer.write_distribution(Distribution(phi_weights.detach()), "phi_weights")
            writer.write_distribution(Distribution(critic_targets.detach()), "critic_targets")
        if self._config.scheduler_config is not None:
            self._actor_scheduler.step()
            self._critic_scheduler.step()

        # train adversarial agent and train with inverted reward r = -r
        batch.rewards = [-r for r in batch.rewards]
        values = self._net.adversarial_critic(inputs=batch.observations, train=False).squeeze().detach()
        phi_weights = self._calculate_phi(batch, values).to(self._device).squeeze(-1).detach()
        critic_targets = get_reward_to_go(batch).to(self._device) if self._config.phi_key != 'gae' else \
            (values + phi_weights).detach()
        critic_loss_distribution = self._train_adversarial_critic_clipped(batch, critic_targets, values)
        actor_loss_distribution = self._train_adversarial_actor_ppo(batch, phi_weights, writer)
        if writer is not None:
            writer.write_distribution(critic_loss_distribution, "adversarial_critic_loss")
            writer.write_distribution(Distribution(phi_weights.detach()), "adversarial_phi_weights")
            writer.write_distribution(Distribution(critic_targets.detach()), "adversarial_critic_targets")
        if self._config.scheduler_config is not None:
            self._adversarial_actor_scheduler.step()
            self._adversarial_critic_scheduler.step()

        self._net.global_step += 1
        self.put_model_back_to_original_device()
        return f" training policy loss {actor_loss_distribution.mean: 0.3e} [{actor_loss_distribution.std: 0.2e}], " \
               f"critic loss {critic_loss_distribution.mean: 0.3e} [{critic_loss_distribution.std: 0.3e}]"
