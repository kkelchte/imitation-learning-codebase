import copy
import os
import time
from typing import List

from src.condor.condor_job import CondorJobConfig, CondorJob, create_jobs_from_job_config_files
from src.condor.helper_functions import create_configs, Dag, translate_keys_to_string
from src.condor.preparation_functions.il_preparation_functions import prepare_default
from src.core.utils import get_date_time_tag


def prepare_domain_adaptation(base_config_file: str,
                              job_config_object: CondorJobConfig,
                              number_of_jobs: int,
                              output_path: str) -> List[CondorJob]:

    domain_adaptation_criterions = ['MMDLossSimple', 'Coral', 'MMDLossZhao']
    epsilons = [0.5, 0.05]
    learning_rates = [0.001, 0.0001]
    model_paths = [os.path.join(output_path, 'models', f'{dac}', f'eps_{eps}', f'lr_{lr}')
                   for dac in domain_adaptation_criterions
                   for eps in epsilons
                   for lr in learning_rates]

    adjustments = {translate_keys_to_string(['output_path']): model_paths,
                   translate_keys_to_string(['trainer_config', 'learning_rate']):
                       [lr for dac in domain_adaptation_criterions
                        for eps in epsilons
                        for lr in learning_rates],
                   translate_keys_to_string(['trainer_config', 'domain_adaptation_criterion']):
                       [dac for dac in domain_adaptation_criterions
                        for eps in epsilons
                        for lr in learning_rates],
                   translate_keys_to_string(['trainer_config', 'epsilon']):
                       [eps for dac in domain_adaptation_criterions
                        for eps in epsilons
                        for lr in learning_rates]}
    config_files = create_configs(base_config=base_config_file,
                                  output_path=output_path,
                                  adjustments=adjustments)
    return create_jobs_from_job_config_files(config_files,
                                             job_config_object=job_config_object)


def prepare_wd_confidence_line_world(base_config_file: str,
                                     job_config_object: CondorJobConfig,
                                     number_of_jobs: int,
                                     output_path: str) -> List[CondorJob]:
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    #weight_decays = [0.0, 0.01, 0.001, 0.0001, ]
    weight_decays = [10, 1.0, 0.1]

    architectures = ['auto_encoder_deeply_supervised_confidence']

    model_paths = [os.path.join(output_path, 'models', arch, f'lr_{lr}', f'wd_{wd}')
                   for arch in architectures
                   for lr in learning_rates
                   for wd in weight_decays]
    adjustments = {translate_keys_to_string(['output_path']): model_paths,
                   translate_keys_to_string(['trainer_config', 'learning_rate']):
                   [lr for arch in architectures
                   for lr in learning_rates
                   for wd in weight_decays],
                   translate_keys_to_string(['architecture_config', 'architecture']):
                   [arch for arch in architectures
                   for lr in learning_rates
                   for wd in weight_decays],
                   translate_keys_to_string(['trainer_config', 'factory_key']):
                       ['DeepSupervisionConfidence' if 'confidence' in arch
                        else 'DeepSupervision'
                        for arch in architectures
                        for lr in learning_rates
                        for wd in weight_decays],
                   translate_keys_to_string(['trainer_config', 'confidence_weight']):
                       [wd for arch in architectures
                        for lr in learning_rates
                        for wd in weight_decays]}
    config_files = create_configs(base_config=base_config_file,
                                  output_path=output_path,
                                  adjustments=adjustments)
    return create_jobs_from_job_config_files(config_files,
                                             job_config_object=job_config_object)


def prepare_lr_pretrain_imagenet(base_config_file: str,
                                 job_config_object: CondorJobConfig,
                                 number_of_jobs: int,
                                 output_path: str) -> List[CondorJob]:
    learning_rates = [0.01, 0.001, 0.0001, 0.00001]
    jobs = []
    for rate in learning_rates:
        job_config = copy.deepcopy(job_config_object)
        job_config.output_path = output_path
        job_config.command += f' -o {output_path}/models/{rate} --learning_rate {rate} ' \
                              f'-c {output_path}/models/{rate}/imagenet_checkpoints/checkpoint_latest.ckpt'
        condor_job = CondorJob(config=job_config)
        condor_job.write_job_file()
        condor_job.write_executable_file()
        jobs.append(condor_job)
    return jobs


def prepare_lr_discirminator_line_world(base_config_file: str,
                                        job_config_object: CondorJobConfig,
                                        number_of_jobs: int,
                                        output_path: str) -> List[CondorJob]:
    learning_rates = [0.001, 0.0001]
    critic_learning_rates = [0.01, 0.001, 0.0001]
    epsilon = [0.1, 0.5, 0.9]
    bns = [False]

    model_paths = [os.path.join(output_path, 'models', 'auto_encoder_deeply_supervised_with_discriminator',
                                f'model_lr_{lr}', f'discriminator_lr_{dlr}', f'epsilon_{eps}', 'bn' if bn else 'default')
                   for lr in learning_rates
                   for dlr in critic_learning_rates
                   for eps in epsilon
                   for bn in bns]
    adjustments = {translate_keys_to_string(['output_path']): model_paths,
                   translate_keys_to_string(['trainer_config', 'learning_rate']):
                   [lr for lr in learning_rates
                   for dlr in critic_learning_rates
                   for eps in epsilon
                    for bn in bns],
                   translate_keys_to_string(['trainer_config', 'critic_learning_rate']):
                   [dlr for lr in learning_rates
                   for dlr in critic_learning_rates
                   for eps in epsilon
                    for bn in bns],
                   translate_keys_to_string(['trainer_config', 'epsilon']):
                   [eps for lr in learning_rates
                   for dlr in critic_learning_rates
                   for eps in epsilon
                    for bn in bns]
                   }
    config_files = create_configs(base_config=base_config_file,
                                  output_path=output_path,
                                  adjustments=adjustments)
    return create_jobs_from_job_config_files(config_files,
                                             job_config_object=job_config_object)


def prepare_lr_architecture_line_world(base_config_file: str,
                                       job_config_object: CondorJobConfig,
                                       number_of_jobs: int,
                                       output_path: str) -> List[CondorJob]:
    learning_rates = [0.001, 0.0001]

    architectures = ['bc_deeply_supervised_auto_encoder']
    # architectures = ['bc_deeply_supervised_auto_encoder',
    #                  'auto_encoder_deeply_supervised_share_weights']

    model_paths = [os.path.join(output_path, 'models', arch, f'lr_{lr}', )
                   for arch in architectures
                   for lr in learning_rates]
    adjustments = {translate_keys_to_string(['output_path']): model_paths,
                   translate_keys_to_string(['trainer_config', 'learning_rate']):
                   [lr for arch in architectures
                   for lr in learning_rates],
                   translate_keys_to_string(['architecture_config', 'architecture']):
                   [arch for arch in architectures
                   for lr in learning_rates],
                   translate_keys_to_string(['trainer_config', 'factory_key']):
                       ['DeepSupervisionConfidence' if 'confidence' in arch
                        else 'DeepSupervision'
                        for arch in architectures
                        for lr in learning_rates],
                   }
    config_files = create_configs(base_config=base_config_file,
                                  output_path=output_path,
                                  adjustments=adjustments)
    return create_jobs_from_job_config_files(config_files,
                                             job_config_object=job_config_object)


def prepare_lr_wd_bn_line_world(base_config_file: str,
                                job_config_object: CondorJobConfig,
                                number_of_jobs: int,
                                output_path: str) -> List[CondorJob]:

    learning_rates = [0.001, 0.0001]
    batch_norm = [True, False]
    weight_decay = [0.001, 0.0001, 0.00001]

    model_paths = [os.path.join(output_path, 'models', 'bn' if bn else 'default', f'wd_{wd}', f'lr_{lr}')
                   for lr in learning_rates
                   for wd in weight_decay
                   for bn in batch_norm]

    adjustments = {translate_keys_to_string(['output_path']): model_paths,
                   translate_keys_to_string(['trainer_config', 'learning_rate']):
                       [lr for lr in learning_rates
                        for wd in weight_decay
                        for bn in batch_norm],
                   translate_keys_to_string(['architecture_config', 'batch_normalisation']):
                       [bn for lr in learning_rates
                        for wd in weight_decay
                        for bn in batch_norm],
                   translate_keys_to_string(['trainer_config', 'weight_decay']):
                       [wd for lr in learning_rates
                        for wd in weight_decay
                        for bn in batch_norm],
                   }
    config_files = create_configs(base_config=base_config_file,
                                  output_path=output_path,
                                  adjustments=adjustments)
    return create_jobs_from_job_config_files(config_files,
                                             job_config_object=job_config_object)


def prepare_data_collection_line_world(base_config_file: str,
                                       job_config_object: CondorJobConfig,
                                       number_of_jobs: int,
                                       output_path: str) -> List[CondorJob]:

    config_files = create_configs(base_config=base_config_file,
                                  output_path=output_path,
                                  adjustments={
                                      '[\"data_saver_config\"][\"saving_directory_tag\"]':
                                          [f'runner_{i}' for i in range(number_of_jobs)],
                                      translate_keys_to_string(['environment_config',
                                                                'ros_config',
                                                                'ros_launch_config',
                                                                'world_name']):
                                          [f'line_worlds/model_{(750 + i):03d}' for i in range(number_of_jobs)],
                                  })
    return create_jobs_from_job_config_files(job_config_files=config_files,
                                             job_config_object=job_config_object)


def prepare_evaluate_interactive_line_world(base_config_file: str,
                                            job_config_object: CondorJobConfig,
                                            number_of_jobs: int,
                                            output_path: str,
                                            model_directories: List[str] = None) -> List[CondorJob]:
    if number_of_jobs == 0:
        return []
    model_directories = [os.path.join(output_path, 'models', d)
                         for d in os.listdir(os.path.join(output_path, 'models'))] \
        if model_directories is None else model_directories

    # evaluate all models in 'number_of_jobs' line worlds
    worlds = [f'line_worlds/model_{i:03d}' for i in range(900, 900 + number_of_jobs)
              for _ in range(len(model_directories))]

    config_files = create_configs(base_config=base_config_file,
                                  output_path=output_path,
                                  adjustments={
                                      '[\"data_saver_config\"][\"saving_directory_tag\"]':
                                          [f'{os.path.basename(d)}_{i}' for i in range(900, 900 + number_of_jobs)
                                           for d in model_directories],
                                      '[\"load_checkpoint_dir\"]':
                                          model_directories * number_of_jobs,
                                      translate_keys_to_string(['environment_config',
                                                                'ros_config',
                                                                'ros_launch_config',
                                                                'world_name']): worlds
                                  })
    return create_jobs_from_job_config_files(job_config_files=config_files,
                                             job_config_object=job_config_object)


def prepare_dag_data_collection_train_evaluate_line_world(base_config_files: List[str],
                                                          job_configs: List[CondorJobConfig],
                                                          number_of_jobs: List[int],
                                                          output_path: str) -> Dag:
    jobs = []
    jobs.extend(prepare_data_collection_line_world(base_config_file=base_config_files[0],
                                                   job_config_object=job_configs[0],
                                                   number_of_jobs=number_of_jobs[0],
                                                   output_path=output_path))
    jobs.extend(prepare_default(base_config_file=base_config_files[1],
                                job_config_object=job_configs[1],
                                number_of_jobs=number_of_jobs[1],
                                output_path=output_path))

    # Add train jobs
    seeds = [123 * n + 5100 for n in range(number_of_jobs[2])]
    model_paths = [os.path.join(output_path, 'models', f'seed_{seed}') for seed in seeds]
    config_files = create_configs(base_config=base_config_files[2],
                                  output_path=output_path,
                                  adjustments={
                                      '[\"architecture_config\"][\"random_seed\"]': seeds,
                                      '[\"output_path\"]': model_paths,
                                  })
    jobs.extend(create_jobs_from_job_config_files(job_config_files=config_files,
                                                  job_config_object=job_configs[2]))
    # Add evaluate jobs
    jobs.extend(prepare_evaluate_interactive_line_world(base_config_file=base_config_files[3],
                                                        job_config_object=job_configs[3],
                                                        number_of_jobs=number_of_jobs[3],
                                                        output_path=output_path,
                                                        model_directories=model_paths))

    dag_lines = '# Prepare_dag_data_collection_train_evaluate: \n'
    # Define jobs:
    start_index = 0
    end_index = number_of_jobs[0]
    for index, job in enumerate(jobs[start_index: end_index]):
        dag_lines += f'JOB data_collection_{index} {job.job_file} \n'
    start_index = number_of_jobs[0]
    end_index = sum(number_of_jobs[0:2])
    assert end_index - start_index == 1
    for index, job in enumerate(jobs[start_index: end_index]):
        dag_lines += f'JOB data_cleaning {job.job_file} \n'
    start_index = sum(number_of_jobs[0:2])
    end_index = sum(number_of_jobs[0:3])
    for index, job in enumerate(jobs[start_index: end_index]):
        dag_lines += f'JOB training_{index} {job.job_file} \n'

    start_index = sum(number_of_jobs[0:3])
    end_index = number_of_jobs[3] * number_of_jobs[2] + start_index
    for index, job in enumerate(jobs[start_index: end_index]):
        dag_lines += f'JOB evaluation_{index} {job.job_file} \n'

    # Define links:
    dag_lines += f'PARENT {" ".join([f"data_collection_{i}" for i in range(number_of_jobs[0])])}' \
                 f' CHILD data_cleaning \n'
    dag_lines += f'PARENT data_cleaning' \
                 f' CHILD {" ".join([f"training_{i}" for i in range(number_of_jobs[2])])} \n'

    for model_index in range(number_of_jobs[2]):
        for world_index in range(0, number_of_jobs[3] * number_of_jobs[2], number_of_jobs[2]):
            dag_lines += f'PARENT training_{model_index} CHILD evaluation_{world_index + model_index} \n'
    # Define retry numbers
    for index in range(number_of_jobs[0]):
        dag_lines += f'Retry data_collection_{index} 2 \n'
    dag_lines += f'Retry data_cleaning 3 \n'
    for index, job in enumerate(jobs[:number_of_jobs[2]]):
        dag_lines += f'Retry training_{index} 2 \n'
    for index, job in enumerate(jobs[:number_of_jobs[3] * number_of_jobs[2]]):
        dag_lines += f'Retry evaluation_{index} 3 \n'
    # Create DAG object
    return Dag(lines_dag_file=dag_lines,
               dag_directory=os.path.join(output_path, 'dag', get_date_time_tag()))


def prepare_train_task_decoders(base_config_file: str,
                                job_config_object: CondorJobConfig,
                                number_of_jobs: int,
                                output_path: str) -> List[CondorJob]:
    job_config = job_config_object
    os.makedirs(output_path if output_path.startswith('/') else os.path.join(os.environ['DATADIR'], output_path), 
                exist_ok=True)

    jobs = []
    # tasks = ['autoencoding', 'depth_euclidean', 'jigsaw', 'reshading', 'colorization', 'edge_occlusion', 'keypoints2d',
    #          'room_layout', 'curvature', 'edge_texture', 'keypoints3d', 'segment_unsup2d',
    #          'class_object', 'egomotion', 'nonfixated_pose', 'segment_unsup25d', 'class_scene', 'fixated_pose',
    #          'normal', 'segment_semantic', 'denoising', 'inpainting', 'point_matching', 'vanishing_point']
    
    tasks = ['keypoints3d']
    dataset = 'vanilla'  # 'noisy_augmented'
    learning_rates = [0.1, 0.01, 0.001, 0.0001]

    not_working_models = ['colorization', 'reshading']
    batch_size = 64
    training_epochs = 150

    for i in range(number_of_jobs):
        for mode in ['default', 'end_to_end', 'side_tuning']:
            for task in tasks:
                if task in not_working_models:
                    continue
                for learning_rate in learning_rates:
                    job_output_path = f'{output_path}/{mode}/{task}/{learning_rate}' if i == 0 \
                        else f'{output_path}/{task}/{learning_rate}_{i}'
                    job_config.command = f'python3.8 src/scripts/train_decoders_on_pretrained_features.py ' \
                                         f'--output_path {job_output_path} ' \
                                         f'--learning_rate {learning_rate} ' \
                                         f'--task {task} ' \
                                         f'--dataset {dataset} ' \
                                         f'--datadir /gluster/visics/kkelchte ' \
                                         f'--batch_size {batch_size} --training_epochs {training_epochs}'
                    if mode != 'default':
                        job_config.command += f' --{mode}'
                    condor_job = CondorJob(config=job_config)
                    condor_job.write_job_file()
                    condor_job.write_executable_file()
                    jobs.append(condor_job)
    return jobs
