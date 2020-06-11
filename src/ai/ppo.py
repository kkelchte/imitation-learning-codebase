#!/usr/bin/python3.8
import torch
from torch import nn
import numpy as np

from src.ai.base_net import BaseNet
from src.ai.trainer import TrainerConfig
from src.ai.utils import get_reward_to_go
from src.ai.vpg import VanillaPolicyGradient
from src.data.utils import select
from src.core.data_types import Distribution, Dataset
from src.core.logger import get_logger, cprint
from src.core.tensorboard_wrapper import TensorboardWrapper
from src.core.utils import get_filename_without_extension

"""Given model, config, data_loader, trains a model and logs relevant training information

If later more complex algorithms for RL are to be implemented, they should inherent from here.
Allows combination of outputs as weighted sum in one big backward pass.
"""


class ProximatePolicyGradient(VanillaPolicyGradient):

    def __init__(self, config: TrainerConfig, network: BaseNet):
        super().__init__(config, network, quiet=True)

        self._config.ppo_epsilon = 0.2 if self._config.ppo_epsilon == "default" else self._config.ppo_epsilon
        self._config.kl_target = 0.01 if self._config.kl_target == "default" else self._config.kl_target
        if self._config.max_actor_training_iterations == "default":
            self._config.max_actor_training_iterations = 10
        if self._config.max_critic_training_iterations == "default":
            self._config.max_critic_training_iterations = 10

        self._logger = get_logger(name=get_filename_without_extension(__file__),
                                  output_path=config.output_path,
                                  quiet=True)
        cprint(f'Started.', self._logger)

    def _train_actor_ppo(self, batch: Dataset, phi_weights: torch.Tensor,
                         original_log_probabilities: torch.Tensor, writer: TensorboardWrapper = None) -> Distribution:
        list_batch_loss = []
        list_entropy_loss = []
        for _ in range(self._config.max_actor_training_iterations):
            selected_indices = np.random.choice(list(range(len(batch))),
                                                size=self._config.data_loader_config.batch_size) \
                    if self._config.data_loader_config.batch_size != -1 else list(range(len(batch)))

            mini_batch_observations = select(batch.observations, selected_indices)
            mini_batch_actions = select(batch.actions, selected_indices)
            mini_batch_original_log_probabilities = select(original_log_probabilities, selected_indices)
            mini_batch_phi_weights = select(phi_weights, selected_indices)

            self._actor_optimizer.zero_grad()
            new_log_probabilities = self._net.policy_log_probabilities(inputs=mini_batch_observations,
                                                                       actions=mini_batch_actions,
                                                                       train=True)
            entropy_loss = self._config.entropy_coefficient * \
                self._net.get_policy_entropy(torch.stack(mini_batch_observations).
                                             type(torch.float32), train=True).mean()
            ratio = torch.exp(new_log_probabilities - mini_batch_original_log_probabilities)
            batch_loss = -(torch.min(ratio * mini_batch_phi_weights,
                                     ratio.clamp(1 - self._config.ppo_epsilon,
                                                 1 + self._config.ppo_epsilon) * mini_batch_phi_weights).mean()
                           + entropy_loss)

            kl_approximation = (mini_batch_original_log_probabilities - new_log_probabilities).mean().item()

            if kl_approximation > 1.5 * self._config.kl_target:
                break
            batch_loss.backward()
            if self._config.gradient_clip_norm != -1:
                nn.utils.clip_grad_norm_(self._net.get_actor_parameters(),
                                         self._config.gradient_clip_norm)
            self._actor_optimizer.step()
            list_batch_loss.append(batch_loss.detach())
            list_entropy_loss.append(entropy_loss.detach())
        actor_loss_distribution = Distribution(torch.stack(list_batch_loss))
        if writer is not None:
            writer.set_step(self._net.global_step)
            writer.write_distribution(actor_loss_distribution, "policy_loss")
            writer.write_distribution(Distribution(torch.stack(list_entropy_loss)), "policy_entropy_loss")
            writer.write_scalar(kl_approximation, 'kl_difference')
        return actor_loss_distribution

    def _train_critic(self, batch: Dataset, targets: torch.Tensor) -> Distribution:
        critic_loss = []
        for value_train_it in range(self._config.max_critic_training_iterations):
            selected_indices = np.random.choice(list(range(len(batch))),
                                                size=self._config.data_loader_config.batch_size) \
                if self._config.data_loader_config.batch_size != -1 else list(range(len(batch)))
            critic_loss.append(super()._train_critic(select(batch, selected_indices),
                                                     select(targets, selected_indices)))
        return Distribution(torch.stack(critic_loss))

    def _train_critic_clipped(self, batch: Dataset, targets: torch.Tensor) -> Distribution:
        critic_loss = []
        previous_values = self._net.critic(inputs=batch.observations, train=True).squeeze().detach()
        for value_train_it in range(self._config.max_critic_training_iterations):
            selected_indices = np.random.choice(list(range(len(batch))),
                                                size=self._config.data_loader_config.batch_size) \
                if self._config.data_loader_config.batch_size != -1 else list(range(len(batch)))
            mini_batch_observations = select(batch.observations, selected_indices)
            mini_batch_previous_values = select(previous_values, selected_indices)
            mini_batch_targets = select(targets, selected_indices)
            batch_values = self._net.critic(inputs=mini_batch_observations, train=True).squeeze()
            batch_loss = self._criterion(batch_values, mini_batch_targets)
            # absolute clipping
            clipped_values = mini_batch_previous_values + \
                             (batch_values - mini_batch_previous_values).clamp(-self._config.ppo_epsilon,
                                                                               self._config.ppo_epsilon)
            clipped_loss = self._criterion(clipped_values, mini_batch_targets)
            batch_loss = torch.max(batch_loss, clipped_loss)
            batch_loss.mean().backward()
            if self._config.gradient_clip_norm != -1:
                nn.utils.clip_grad_norm_(self._net.get_critic_parameters(),
                                         self._config.gradient_clip_norm)
            self._critic_optimizer.step()
            critic_loss.append(batch_loss.detach())
        return Distribution(torch.stack(critic_loss))

    def train(self, epoch: int = -1, writer=None) -> str:
        self.put_model_on_device()
        batch = self.data_loader.get_dataset()
        assert len(batch) != 0

        phi_weights = self._calculate_phi(batch).to(self._device)
        original_log_probabilities = self._net.policy_log_probabilities(inputs=batch.observations,
                                                                        actions=batch.actions,
                                                                        train=False).detach()
        actor_loss_distribution = self._train_actor_ppo(batch, phi_weights, original_log_probabilities, writer)
        #        critic_loss_distribution = self._train_critic(batch, get_reward_to_go(batch).to(self._device))
        # In case of advantages, use current value estimate with advantage to get target estimate
        critic_targets = get_reward_to_go(batch).to(self._device) if self._config.phi_key != 'gae' else \
            self._net.critic(inputs=batch.observations, train=True).squeeze().detach() + phi_weights
        critic_loss_distribution = self._train_critic_clipped(batch, critic_targets)

        if writer is not None:
            writer.write_distribution(critic_loss_distribution, "critic_loss")
            writer.write_distribution(Distribution(phi_weights.detach()), "phi_weights")
            writer.write_distribution(Distribution(critic_targets.detach()), "critic_targets")

        if self._actor_scheduler is not None:
            self._actor_scheduler.step()
        if self._critic_scheduler is not None:
            self._critic_scheduler.step()
        self._net.global_step += 1
        self._save_checkpoint(epoch=epoch)
        self.put_model_back_to_original_device()
        return f" training policy loss {actor_loss_distribution.mean: 0.3e} [{actor_loss_distribution.std: 0.2e}], " \
               f"critic loss {critic_loss_distribution.mean: 0.3e} [{critic_loss_distribution.std: 0.3e}]"
