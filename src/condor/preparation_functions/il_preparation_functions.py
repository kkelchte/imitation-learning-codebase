import os
from typing import List

from src.condor.condor_job import CondorJobConfig, CondorJob, create_jobs_from_job_config_files
from src.condor.helper_functions import create_configs, Dag
from src.core.utils import get_date_time_tag


def prepare_default(base_config_file: str,
                    job_config_object: CondorJobConfig,
                    number_of_jobs: int,
                    output_path: str) -> List[CondorJob]:
    """Launch number of condor_jobs performing script with base_config"""
    if number_of_jobs == 0:
        return []
    default_configs = create_configs(base_config=base_config_file,
                                     output_path=output_path,
                                     adjustments={
                                        '[\"output_path\"]':
                                        [output_path+'_'+str(i) for i in range(number_of_jobs)],
                                     } if number_of_jobs > 1 else {})
    return create_jobs_from_job_config_files(default_configs,
                                             job_config_object=job_config_object)


def prepare_data_collection(base_config_file: str,
                            job_config_object: CondorJobConfig,
                            number_of_jobs: int,
                            output_path: str) -> List[CondorJob]:
    config_files = create_configs(base_config=base_config_file,
                                  output_path=output_path,
                                  adjustments={
                                      '[\"data_saver_config\"][\"saving_directory_tag\"]':
                                          [f'runner_{i}' for i in range(number_of_jobs)]
                                  })
    return create_jobs_from_job_config_files(job_config_files=config_files,
                                             job_config_object=job_config_object)


def prepare_train(base_config_file: str,
                  job_config_object: CondorJobConfig,
                  number_of_jobs: int,
                  output_path: str) -> List[CondorJob]:
    if number_of_jobs == 0:
        return []
    seeds = [123 * n + 5100 for n in range(number_of_jobs)]
    model_paths = [os.path.join(output_path, 'models', f'seed_{seed}') for seed in seeds]
    config_files = create_configs(base_config=base_config_file,
                                  output_path=output_path,
                                  adjustments={
                                      '[\"architecture_config\"][\"random_seed\"]': seeds,
                                      '[\"output_path\"]': model_paths,
                                  })
    return create_jobs_from_job_config_files(job_config_files=config_files,
                                             job_config_object=job_config_object)


def prepare_evaluate_interactive(base_config_file: str,
                                 job_config_object: CondorJobConfig,
                                 number_of_jobs: int,
                                 output_path: str,
                                 model_directories: List[str] = None) -> List[CondorJob]:
    if number_of_jobs == 0:
        return []
    model_directories = [os.path.join(output_path, 'models', d)
                         for d in os.listdir(os.path.join(output_path, 'models'))] \
        if model_directories is None else model_directories

    model_directories = model_directories[-min(number_of_jobs, len(model_directories)):]
    config_files = create_configs(base_config=base_config_file,
                                  output_path=output_path,
                                  adjustments={
                                      '[\"data_saver_config\"][\"saving_directory_tag\"]':
                                          [os.path.basename(d) for d in model_directories],
                                      '[\"load_checkpoint_dir\"]':
                                          model_directories,
                                  })
    return create_jobs_from_job_config_files(job_config_files=config_files,
                                             job_config_object=job_config_object)


def prepare_dag_data_collection(base_config_files: List[str],
                                job_configs: List[CondorJobConfig],
                                number_of_jobs: List[int],
                                output_path: str) -> Dag:
    jobs = []
    jobs.extend(prepare_data_collection(base_config_file=base_config_files[0],
                                        job_config_object=job_configs[0],
                                        number_of_jobs=number_of_jobs[0],
                                        output_path=output_path))
    jobs.extend(prepare_default(base_config_file=base_config_files[1],
                                job_config_object=job_configs[1],
                                number_of_jobs=number_of_jobs[1],
                                output_path=output_path))
    dag_lines = '# Prepare_dag_data_collection: \n'
    for index, job in enumerate(jobs[:-1]):
        dag_lines += f'JOB data_collection_{index} {job.job_file} \n'
    num_collection_jobs = len(jobs) - 1
    dag_lines += f'JOB data_cleaning {jobs[-1].job_file} \n'
    dag_lines += f'PARENT {" ".join([f"data_collection_{index}" for index in range(num_collection_jobs)])} ' \
                 f'CHILD data_cleaning \n'
    for index in range(num_collection_jobs):
        dag_lines += f'Retry data_collection_{index} 2 \n'
    dag_lines += f'Retry data_cleaning 3 \n'
    return Dag(lines_dag_file=dag_lines,
               dag_directory=os.path.join(output_path, 'dag', get_date_time_tag()))


def prepare_dag_train_evaluate(base_config_files: List[str],
                               job_configs: List[CondorJobConfig],
                               number_of_jobs: List[int],
                               output_path: str) -> Dag:
    jobs = []
    # Add train jobs
    seeds = [123 * n + 5100 for n in range(number_of_jobs[0])]
    model_paths = [os.path.join(output_path, 'models', f'seed_{seed}') for seed in seeds]
    config_files = create_configs(base_config=base_config_files[0],
                                  output_path=output_path,
                                  adjustments={
                                      '[\"architecture_config\"][\"random_seed\"]': seeds,
                                      '[\"output_path\"]': model_paths,
                                  })
    jobs.extend(create_jobs_from_job_config_files(job_config_files=config_files,
                                                  job_config_object=job_configs[0]))
    # Add evaluate jobs
    jobs.extend(prepare_evaluate_interactive(base_config_file=base_config_files[1],
                                             job_config_object=job_configs[1],
                                             number_of_jobs=number_of_jobs[1],
                                             output_path=output_path,
                                             model_directories=model_paths))

    dag_lines = '# prepare_dag_train_evaluate: \n'
    for index, job in enumerate(jobs[:number_of_jobs[0]]):
        dag_lines += f'JOB training_{index} {job.job_file} \n'
    for index, job in enumerate(jobs[number_of_jobs[0]:
                                     number_of_jobs[0] + number_of_jobs[1]]):
        dag_lines += f'JOB evaluation_{index} {job.job_file} \n'

    number_of_links = min(number_of_jobs)
    for index in range(number_of_links):
        dag_lines += f'PARENT training_{index} CHILD evaluation_{index} \n'

    for index, job in enumerate(jobs[:number_of_jobs[0]]):
        dag_lines += f'Retry training_{index} 2 \n'
    for index, job in enumerate(jobs[number_of_jobs[0]:
                                     number_of_jobs[0] + number_of_jobs[1]]):
        dag_lines += f'Retry evaluation_{index} 3 \n'

    return Dag(lines_dag_file=dag_lines,
               dag_directory=os.path.join(output_path, 'dag', get_date_time_tag()))


def prepare_dag_data_collection_train_evaluate(base_config_files: List[str],
                                               job_configs: List[CondorJobConfig],
                                               number_of_jobs: List[int],
                                               output_path: str) -> Dag:
    jobs = []
    jobs.extend(prepare_data_collection(base_config_file=base_config_files[0],
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
    jobs.extend(prepare_evaluate_interactive(base_config_file=base_config_files[3],
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
    end_index = sum(number_of_jobs[:])
    for index, job in enumerate(jobs[start_index: end_index]):
        dag_lines += f'JOB evaluation_{index} {job.job_file} \n'
    # Define links:
    dag_lines += f'PARENT {" ".join([f"data_collection_{i}" for i in range(number_of_jobs[0])])}' \
                 f' CHILD data_cleaning \n'
    dag_lines += f'PARENT data_cleaning' \
                 f' CHILD {" ".join([f"training_{i}" for i in range(number_of_jobs[2])])} \n'
    number_of_links = min(number_of_jobs[2:])
    for index in range(number_of_links):
        dag_lines += f'PARENT training_{index} CHILD evaluation_{index} \n'
    # Define retry numbers
    for index in range(number_of_jobs[0]):
        dag_lines += f'Retry data_collection_{index} 2 \n'
    dag_lines += f'Retry data_cleaning 3 \n'
    for index, job in enumerate(jobs[:number_of_jobs[2]]):
        dag_lines += f'Retry training_{index} 2 \n'
    for index, job in enumerate(jobs[:number_of_jobs[3]]):
        dag_lines += f'Retry evaluation_{index} 3 \n'
    # Create DAG object
    return Dag(lines_dag_file=dag_lines,
               dag_directory=os.path.join(output_path, 'dag', get_date_time_tag()))
