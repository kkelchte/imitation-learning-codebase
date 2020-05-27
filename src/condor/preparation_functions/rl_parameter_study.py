import os
from typing import List

from src.condor.condor_job import CondorJobConfig, CondorJob, create_jobs_from_job_config_files
from src.condor.helper_functions import create_configs, translate_keys_to_string


def prepare_batch_size_study(base_config_file: str,
                             job_config_object: CondorJobConfig,
                             number_of_jobs: int,
                             output_path: str) -> List[CondorJob]:
    batch_sizes = [50, 100, 500, 1000, 5000]
    seeds = [123 * n + 5100 for n in range(number_of_jobs)]
    model_paths = [os.path.join(output_path, 'models', f'sd_{seed}_bs_{bs}') for bs in batch_sizes for seed in seeds]
    adjustments = {translate_keys_to_string(['architecture_config',
                                            'initialisation_seed']): seeds * len(batch_sizes),
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
    learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    seeds = [123 * n + 5100 for n in range(number_of_jobs)]
    model_paths = [os.path.join(output_path, 'models', f'sd_{seed}_lr_{lr}') for lr in learning_rates for seed in seeds]
    adjustments = {translate_keys_to_string(['architecture_config',
                                            'initialisation_seed']): seeds * len(learning_rates),
                   translate_keys_to_string(['output_path']): model_paths,
                   translate_keys_to_string(['trainer_config', 'learning_rate']):
                       [bs for bs in learning_rates for _ in range(len(seeds))]}
    config_files = create_configs(base_config=base_config_file,
                                  output_path=output_path,
                                  adjustments=adjustments)
    return create_jobs_from_job_config_files(config_files,
                                             job_config_object=job_config_object)


def prepare_optimiser_study(base_config_file: str,
                            job_config_object: CondorJobConfig,
                            number_of_jobs: int,
                            output_path: str) -> List[CondorJob]:
    optimisers = ['SGD', 'Adam', 'Adadelta', 'RMSprop']
    seeds = [123 * n + 5100 for n in range(number_of_jobs)]
    model_paths = [os.path.join(output_path, 'models', f'sd_{seed}_opt_{opt}') for opt in optimisers for seed in seeds]
    adjustments = {translate_keys_to_string(['architecture_config',
                                            'initialisation_seed']): seeds * len(optimisers),
                   translate_keys_to_string(['output_path']): model_paths,
                   translate_keys_to_string(['trainer_config', 'learning_rate']):
                       [bs for bs in optimisers for _ in range(len(seeds))]}
    config_files = create_configs(base_config=base_config_file,
                                  output_path=output_path,
                                  adjustments=adjustments)
    return create_jobs_from_job_config_files(config_files,
                                             job_config_object=job_config_object)
