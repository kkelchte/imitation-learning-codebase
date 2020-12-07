import os
from typing import List

from src.condor.condor_job import CondorJobConfig, CondorJob, create_jobs_from_job_config_files
from src.condor.helper_functions import create_configs, translate_keys_to_string


def prepare_param_study(base_config_file: str,
                        job_config_object: CondorJobConfig,
                        number_of_jobs: int,
                        output_path: str) -> List[CondorJob]:
    seeds = [132 * n + 5100 for n in range(number_of_jobs)]
    model_paths = [os.path.join(output_path, f'sd_{seed}') for seed in seeds]
    adjustments = {translate_keys_to_string(['architecture_config',
                                             'random_seed']): [seed for seed in seeds],
                   translate_keys_to_string(['output_path']): model_paths}
    config_files = create_configs(base_config=base_config_file,
                                  output_path=output_path,
                                  adjustments=adjustments)
    return create_jobs_from_job_config_files(config_files,
                                             job_config_object=job_config_object)


def prepare_architecture_study(base_config_file: str,
                               job_config_object: CondorJobConfig,
                               number_of_jobs: int,
                               output_path: str) -> List[CondorJob]:
    seeds = [123 * n + 5100 for n in range(number_of_jobs)]
    architectures = ['adversarial_actor_critic', 'fleeing_actor_critic', 'tracking_actor_critic']
    model_paths = [os.path.join(output_path, 'models', a, f'sd_{seed}') for a in architectures for seed in seeds]
    adjustments = {translate_keys_to_string(['architecture_config',
                                             'random_seed']): [seed for a in architectures for seed in seeds],
                   translate_keys_to_string(['architecture_config',
                                             'architecture']): [a for a in architectures for seed in seeds],
                   translate_keys_to_string(['output_path']): model_paths,
                   translate_keys_to_string(['trainer_config', 'factory_key']):
                       ['APPO' if 'adversarial' in a else 'PPO' for a in architectures for seed in seeds]}
    config_files = create_configs(base_config=base_config_file,
                                  output_path=output_path,
                                  adjustments=adjustments)
    return create_jobs_from_job_config_files(config_files,
                                             job_config_object=job_config_object)


def prepare_batch_size_study(base_config_file: str,
                             job_config_object: CondorJobConfig,
                             number_of_jobs: int,
                             output_path: str) -> List[CondorJob]:
    batch_sizes = [50, 100, 500, 1000, 5000]
    seeds = [123 * n + 5961 for n in range(number_of_jobs)]
    model_paths = [os.path.join(output_path, 'models', f'sd_{seed}/bs_{bs}') for bs in batch_sizes for seed in seeds]
    adjustments = {translate_keys_to_string(['architecture_config',
                                             'random_seed']): seeds * len(batch_sizes),
                   translate_keys_to_string(['output_path']): model_paths,
                   translate_keys_to_string(['trainer_config', 'data_loader_config', 'batch_size']):
                       [bs for bs in batch_sizes for _ in range(len(seeds))]}
    config_files = create_configs(base_config=base_config_file,
                                  output_path=output_path,
                                  adjustments=adjustments)
    return create_jobs_from_job_config_files(config_files,
                                             job_config_object=job_config_object)


def prepare_learning_rate_study(base_config_file: str,
                                job_config_object: CondorJobConfig,
                                number_of_jobs: int,
                                output_path: str) -> List[CondorJob]:
    learning_rates = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
    seeds = [123 * n + 5961 for n in range(number_of_jobs)]
    model_paths = [os.path.join(output_path, 'models', f'sd_{seed}/lr_{lr}') for lr in learning_rates for seed in seeds]
    adjustments = {translate_keys_to_string(['architecture_config',
                                             'random_seed']): seeds * len(learning_rates),
                   translate_keys_to_string(['output_path']): model_paths,
                   translate_keys_to_string(['trainer_config', 'learning_rate']):
                       [bs for bs in learning_rates for _ in range(len(seeds))],
                   translate_keys_to_string(['trainer_config', 'actor_learning_rate']):
                       [bs for bs in learning_rates for _ in range(len(seeds))],
                   }
    config_files = create_configs(base_config=base_config_file,
                                  output_path=output_path,
                                  adjustments=adjustments)
    return create_jobs_from_job_config_files(config_files,
                                             job_config_object=job_config_object)


def prepare_optimiser_study(base_config_file: str,
                            job_config_object: CondorJobConfig,
                            number_of_jobs: int,
                            output_path: str) -> List[CondorJob]:
    optimizers = ['SGD', 'Adam', 'Adadelta', 'RMSprop']
    seeds = [123 * n + 5961 for n in range(number_of_jobs)]
    model_paths = [os.path.join(output_path, 'models', f'sd_{seed}/opt_{opt}') for opt in optimizers for seed in seeds]
    adjustments = {translate_keys_to_string(['architecture_config',
                                             'random_seed']): seeds * len(optimizers),
                   translate_keys_to_string(['output_path']): model_paths,
                   translate_keys_to_string(['trainer_config', 'optimizer']):
                       [bs for bs in optimizers for _ in range(len(seeds))]}
    config_files = create_configs(base_config=base_config_file,
                                  output_path=output_path,
                                  adjustments=adjustments)
    return create_jobs_from_job_config_files(config_files,
                                             job_config_object=job_config_object)


def prepare_loss_study(base_config_file: str,
                       job_config_object: CondorJobConfig,
                       number_of_jobs: int,
                       output_path: str) -> List[CondorJob]:
    losses = ['MSELoss', 'L1Loss', 'SmoothL1Loss']
    seeds = [123 * n + 5961 for n in range(number_of_jobs)]
    model_paths = [os.path.join(output_path, 'models', f'sd_{seed}/loss_{loss}') for loss in losses for seed in seeds]
    adjustments = {translate_keys_to_string(['architecture_config',
                                             'random_seed']): seeds * len(losses),
                   translate_keys_to_string(['output_path']): model_paths,
                   translate_keys_to_string(['trainer_config', 'criterion']):
                       [bs for bs in losses for _ in range(len(seeds))]}
    config_files = create_configs(base_config=base_config_file,
                                  output_path=output_path,
                                  adjustments=adjustments)
    return create_jobs_from_job_config_files(config_files,
                                             job_config_object=job_config_object)


def prepare_phi_study(base_config_file: str,
                      job_config_object: CondorJobConfig,
                      number_of_jobs: int,
                      output_path: str) -> List[CondorJob]:
    phi_keys = ["gae", "reward-to-go", "return", "value-baseline"]
    seeds = [123 * n + 5961 for n in range(number_of_jobs)]
    model_paths = [os.path.join(output_path, 'models', f'sd_{seed}/phi_{x}') for x in phi_keys for seed in seeds]
    adjustments = {translate_keys_to_string(['architecture_config',
                                             'random_seed']): seeds * len(phi_keys),
                   translate_keys_to_string(['output_path']): model_paths,
                   translate_keys_to_string(['trainer_config', 'phi_key']):
                       [x for x in phi_keys for _ in range(len(seeds))]}
    config_files = create_configs(base_config=base_config_file,
                                  output_path=output_path,
                                  adjustments=adjustments)
    return create_jobs_from_job_config_files(config_files,
                                             job_config_object=job_config_object)


def prepare_ppo_epsilon_study(base_config_file: str,
                              job_config_object: CondorJobConfig,
                              number_of_jobs: int,
                              output_path: str) -> List[CondorJob]:
    ppo_epsilon = [0.02, 0.1, 0.2, 1, 2]
    seeds = [123 * n + 5961 for n in range(number_of_jobs)]
    model_paths = [os.path.join(output_path, 'models', f'sd_{seed}/eps_{x}') for x in ppo_epsilon for seed in seeds]
    adjustments = {translate_keys_to_string(['architecture_config',
                                             'random_seed']): seeds * len(ppo_epsilon),
                   translate_keys_to_string(['output_path']): model_paths,
                   translate_keys_to_string(['trainer_config', 'ppo_epsilon']):
                       [x for x in ppo_epsilon for _ in range(len(seeds))],
                   translate_keys_to_string(['trainer_config', 'factory_key']):
                       ['PPO' for x in ppo_epsilon for _ in range(len(seeds))]}
    config_files = create_configs(base_config=base_config_file,
                                  output_path=output_path,
                                  adjustments=adjustments)
    return create_jobs_from_job_config_files(config_files,
                                             job_config_object=job_config_object)


def prepare_ppo_kl_target_study(base_config_file: str,
                                job_config_object: CondorJobConfig,
                                number_of_jobs: int,
                                output_path: str) -> List[CondorJob]:
    kl_targets = [0.001, 0.005, 0.01, 0.05, 0.1]
    seeds = [123 * n + 5100 for n in range(number_of_jobs)]
    model_paths = [os.path.join(output_path, 'models', f'sd_{seed}/kl_{x}') for x in kl_targets for seed in seeds]
    adjustments = {translate_keys_to_string(['architecture_config',
                                             'random_seed']): seeds * len(kl_targets),
                   translate_keys_to_string(['output_path']): model_paths,
                   translate_keys_to_string(['trainer_config', 'kl_target']):
                       [x for x in kl_targets for _ in range(len(seeds))],
                   translate_keys_to_string(['trainer_config', 'factory_key']):
                       ['PPO' for x in kl_targets for _ in range(len(seeds))]}
    config_files = create_configs(base_config=base_config_file,
                                  output_path=output_path,
                                  adjustments=adjustments)
    return create_jobs_from_job_config_files(config_files,
                                             job_config_object=job_config_object)


def prepare_ppo_max_train_steps_study(base_config_file: str,
                                      job_config_object: CondorJobConfig,
                                      number_of_jobs: int,
                                      output_path: str) -> List[CondorJob]:
    max_value_training_iterations = [1, 5, 10, 50]
    max_actor_training_iterations = [1, 5, 10, 50]
    seeds = [123 * n + 5961 for n in range(number_of_jobs)]
    model_paths = [os.path.join(output_path, 'models', f'sd_{seed}/p_{x}_v_{y}')
                   for y in max_value_training_iterations
                   for x in max_actor_training_iterations
                   for seed in seeds]
    adjustments = {translate_keys_to_string(['architecture_config', 'random_seed']):
                       seeds * len(max_actor_training_iterations) * len(max_value_training_iterations),
                   translate_keys_to_string(['output_path']): model_paths,
                   translate_keys_to_string(['trainer_config', 'max_actor_training_iterations']):
                       [x for _ in max_value_training_iterations
                        for x in max_actor_training_iterations
                        for _ in range(len(seeds))],
                   translate_keys_to_string(['trainer_config', 'max_critic_training_iterations']):
                       [x for x in max_value_training_iterations
                        for _ in max_actor_training_iterations
                        for _ in range(len(seeds))],
                   translate_keys_to_string(['trainer_config', 'factory_key']):
                       ['PPO' for _ in max_value_training_iterations
                        for _ in max_actor_training_iterations for _ in range(len(seeds))]}
    config_files = create_configs(base_config=base_config_file,
                                  output_path=output_path,
                                  adjustments=adjustments)
    return create_jobs_from_job_config_files(config_files,
                                             job_config_object=job_config_object)


def prepare_entropy_study(base_config_file: str,
                          job_config_object: CondorJobConfig,
                          number_of_jobs: int,
                          output_path: str) -> List[CondorJob]:
    entropy_vals = [0.0, 0.1, -0.1]
    seeds = [123 * n + 5961 for n in range(number_of_jobs)]
    model_paths = [os.path.join(output_path, 'models', f'sd_{seed}/entr_{x}') for x in entropy_vals for seed in seeds]
    adjustments = {translate_keys_to_string(['architecture_config',
                                             'random_seed']): seeds * len(entropy_vals),
                   translate_keys_to_string(['output_path']): model_paths,
                   translate_keys_to_string(['trainer_config', 'entropy_coefficient']):
                       [x for x in entropy_vals for _ in range(len(seeds))]}
    config_files = create_configs(base_config=base_config_file,
                                  output_path=output_path,
                                  adjustments=adjustments)
    return create_jobs_from_job_config_files(config_files,
                                             job_config_object=job_config_object)
