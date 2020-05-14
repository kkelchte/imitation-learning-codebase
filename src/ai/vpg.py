#!/usr/bin/python3

import torch

from src.ai.base_net import BaseNet
from src.ai.trainer import Trainer, TrainerConfig
from src.ai.utils import get_returns, get_reward_to_go, get_generalized_advantage_estimate
from src.core.data_types import Dataset
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension

"""Given model, config, data_loader, trains a model and logs relevant training information

If later more complex algorithms for RL are to be implemented, they should inherent from here.
Allows combination of outputs as weighted sum in one big backward pass.
"""


class VanillaPolicyGradient(Trainer):

    def __init__(self, config: TrainerConfig, network: BaseNet, quiet: bool = False):
        super().__init__(config, network, super_init=True)
        # Set default config params
        self._config.phi_key = 'gae' if self._config.phi_key == 'default' else self._config.phi_key
        self._config.discount = 0.95 if self._config.discount == 'default' else self._config.discount
        self._config.gae_lambda = 0.95 if self._config.gae_lambda == 'default' else self._config.gae_lambda

        if not quiet:
            self._logger = get_logger(name=get_filename_without_extension(__file__),
                                      output_path=config.output_path,
                                      quiet=True)
            cprint(f'Started.', self._logger)

        self._actor_optimizer = eval(f'torch.optim.{self._config.optimizer}')(params=self._net.get_actor_parameters(),
                                                                              lr=self._config.learning_rate)
        self._critic_optimizer = eval(f'torch.optim.{self._config.optimizer}')(params=self._net.get_critic_parameters(),
                                                                               lr=self._config.learning_rate)

    def _calculate_phi(self, batch: Dataset) -> torch.Tensor:
        if self._config.phi_key == "return":
            return get_returns(batch)
        elif self._config.phi_key == "reward-to-go":
            return get_reward_to_go(batch)
        elif self._config.phi_key == "gae":
            return get_generalized_advantage_estimate(
                batch_rewards=batch.rewards,
                batch_done=batch.done,
                batch_values=self._net.critic(batch.observations, train=False).detach(),
                discount=0.95 if self._config.discount != "default" else self._config.discount,
                gae_lambda=0.95 if self._config.gae_lambda != "default" else self._config.gae_lambda,
            )
        elif self._config.phi_key == "value-baseline":
            values = self._net.critic(batch.observations, train=False).detach()
            returns = get_reward_to_go(batch)
            return returns - values
        else:
            raise NotImplementedError

    def _train_actor(self, batch: Dataset, phi_weights: torch.Tensor) -> torch.Tensor:
        self._actor_optimizer.zero_grad()
        log_probability = self._net.policy_log_probabilities(torch.stack(batch.observations).type(torch.float32),
                                                             torch.stack(batch.actions).type(torch.float32),
                                                             train=True)
        policy_loss = -(log_probability * phi_weights)
        policy_loss.mean().backward()
        self._actor_optimizer.step()
        return policy_loss.mean().detach()

    def _train_critic(self, batch: Dataset, phi_weights: torch.Tensor) -> torch.Tensor:
        self._critic_optimizer.zero_grad()
        critic_loss = self._criterion(self._net.critic(inputs=batch.observations, train=True).squeeze(), phi_weights)
        # critic_loss = ((self._net.critic(inputs=batch.observations, train=True).squeeze() - phi_weights) ** 2).mean()
        critic_loss.mean().backward()
        self._critic_optimizer.step()
        return critic_loss.mean().detach()

    def train(self, epoch: int = -1, writer=None, phi_weights=None) -> str:
        self.put_model_on_device()
        batch = self.data_loader.get_dataset()
        assert len(batch) != 0

        if phi_weights is None:
            phi_weights = self._calculate_phi(batch)
        policy_loss = self._train_actor(batch, phi_weights)
        critic_loss = self._train_critic(batch, get_reward_to_go(batch))
        self._net.global_step += 1

        self._save_checkpoint(epoch)
        self.put_model_back_to_original_device()

        if writer is not None:
            writer.set_step(self._net.global_step)
            writer.write_scalar(policy_loss.data, "policy_loss")
            writer.write_scalar(critic_loss.data, "critic_loss")
        return f" training policy loss {policy_loss.data: 0.3e}, critic loss {critic_loss.data: 0.3e}"
