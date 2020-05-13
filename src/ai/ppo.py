#!/usr/bin/python3.8
import torch

from src.ai.base_net import BaseNet
from src.ai.trainer import TrainerConfig
from src.ai.vpg import VanillaPolicyGradient
from src.core.data_types import Distribution, Dataset
from src.core.logger import get_logger, cprint
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
            self._config.max_critic_training_iterations = 3

        self._logger = get_logger(name=get_filename_without_extension(__file__),
                                  output_path=config.output_path,
                                  quiet=True)
        cprint(f'Started.', self._logger)

    def _train_actor_ppo(self, batch: Dataset, phi_weights: torch.Tensor, original_log_probabilities: torch.Tensor) \
            -> Distribution:
        actor_loss = []
        for _ in range(self._config.max_actor_training_iterations):
            self._actor_optimizer.zero_grad()
            new_log_probabilities = self._net.policy_log_probabilities(inputs=batch.observations,
                                                                       actions=batch.actions,
                                                                       train=True)

            ratio = torch.exp(new_log_probabilities - original_log_probabilities)
            batch_loss = -torch.min(ratio * phi_weights,
                                    ratio.clamp(1 - self._config.ppo_epsilon,
                                                1 + self._config.ppo_epsilon) * phi_weights).mean()
            kl_approximation = (original_log_probabilities - new_log_probabilities).mean().item()

            if kl_approximation > 1.5 * self._config.kl_target:
                print(f'EARLY BREAK {_}')
                break
            batch_loss.backward()
            self._actor_optimizer.step()
            actor_loss.append(batch_loss.detach())
        return Distribution(torch.stack(actor_loss))

    def _train_critic(self, batch: Dataset, phi_weights: torch.Tensor) -> Distribution:
        critic_loss = []
        for value_train_it in range(self._config.max_critic_training_iterations):
            critic_loss.append(super()._train_critic(batch, phi_weights))
        return Distribution(torch.stack(critic_loss))

    def train(self, epoch: int = -1, writer=None) -> str:
        self.put_model_on_device()
        batch = self.data_loader.get_dataset()
        assert len(batch) != 0

        phi_weights = self._calculate_phi(batch)
        original_log_probabilities = self._net.policy_log_probabilities(inputs=batch.observations,
                                                                        actions=batch.actions,
                                                                        train=False).detach()
        actor_loss_distribution = self._train_actor_ppo(batch, phi_weights, original_log_probabilities)
        critic_loss_distribution = self._train_critic(batch, phi_weights)
        self._net.global_step += 1

        self._save_checkpoint(epoch=epoch)
        self.put_model_back_to_original_device()

        if writer is not None:
            writer.set_step(self._net.global_step)
            writer.write_distribution(actor_loss_distribution, "policy_loss")
            writer.write_distribution(critic_loss_distribution, "critic_loss")
        return f" training policy loss {actor_loss_distribution.mean: 0.3e} [{actor_loss_distribution.std: 0.2e}], " \
               f"critic loss {critic_loss_distribution.mean: 0.3e} [{critic_loss_distribution.std: 0.3e}]"
