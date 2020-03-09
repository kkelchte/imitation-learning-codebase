import copy
import os
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import List

import yaml
from dataclasses_json import dataclass_json

from src.condor.condor_job import CondorJobConfig, CondorJob
from src.condor.helper_functions import create_configs, Dag
from src.core.config_loader import Parser, Config
from src.core.utils import camelcase_to_snake_format, get_date_time_tag


class CondorLauncherMode(IntEnum):
    DataCollection = 0
    TrainModel = 1
    EvaluateModel = 2
    DataCleaning = 3
    DagDataCollection = 4
    DagTrainEvaluate = 5
    DagDataCollectionTrainEvaluate = 6


@dataclass_json
@dataclass
class CondorLauncherConfig(Config):
    mode: CondorLauncherMode = None
    number_of_jobs: List[int] = 1
    base_config_files: List[str] = None
    job_configs: List[CondorJobConfig] = None


class CondorLauncher:

    def __init__(self, config: CondorLauncherConfig):
        self._config = config
        self._jobs: List[CondorJob] = []
        self._model_paths: List[str] = []  # link training and evaluation model paths in train - evaluate dag
        self._dag = None

        self.prepare_factory()

    def launch(self):
        if self._config.mode in [CondorLauncherMode.DataCollection,
                                 CondorLauncherMode.TrainModel,
                                 CondorLauncherMode.EvaluateModel,
                                 CondorLauncherMode.DataCleaning]:
            for job in self._jobs:
                job.submit()
        elif self._config.mode in [CondorLauncherMode.DagDataCollection,
                                   CondorLauncherMode.DagTrainEvaluate,
                                   CondorLauncherMode.DagDataCollectionTrainEvaluate]:
            self._dag.submit()
        else:
            raise NotImplementedError

    def prepare_factory(self):
        eval(f'self.prepare_{camelcase_to_snake_format(self._config.mode.name)}()')

    def create_jobs_from_job_config_files(self,
                                          job_config_files: List[str],
                                          job_config_object: CondorJobConfig = None) -> None:
        job_config_object = self._config.job_configs[0] if job_config_object is None else job_config_object
        for config in job_config_files:
            job_config = copy.deepcopy(job_config_object)
            job_config.config_file = config
            condor_job = CondorJob(config=job_config)
            condor_job.write_job_file()
            condor_job.write_executable_file()
            self._jobs.append(condor_job)
            time.sleep(1)

    def prepare_data_collection(self, base_config_file: str = None, job_config_object: CondorJobConfig = None,
                                number_of_jobs: int = None):
        base_config = self._config.base_config_files[0] if base_config_file is None else base_config_file
        job_config_object = self._config.job_configs[0] if job_config_object is None else job_config_object
        number_of_jobs = self._config.number_of_jobs[0] if number_of_jobs is None else number_of_jobs

        config_files = create_configs(base_config=base_config,
                                      output_path=self._config.output_path,
                                      adjustments={
                                          '[\"data_saver_config\"][\"saving_directory_tag\"]':
                                              list(range(number_of_jobs))
                                      })
        self.create_jobs_from_job_config_files(job_config_files=config_files,
                                               job_config_object=job_config_object)

    def prepare_train_model(self, base_config_file: str = None, job_config_object: CondorJobConfig = None,
                            number_of_jobs: int = None):
        base_config = self._config.base_config_files[0] if base_config_file is None else base_config_file
        job_config_object = self._config.job_configs[0] if job_config_object is None else job_config_object
        number_of_jobs = self._config.number_of_jobs[0] if number_of_jobs is None else number_of_jobs
        if number_of_jobs == 0:
            return
        seeds = [123 * n + 5100 for n in range(number_of_jobs)]
        self._model_paths = [os.path.join(self._config.output_path, 'models', f'seed_{seed}') for seed in seeds]
        config_files = create_configs(base_config=base_config,
                                      output_path=self._config.output_path,
                                      adjustments={
                                          '[\"model_config\"][\"initialisation_seed\"]': seeds,
                                          '[\"generate_new_output_path\"]': [False] * number_of_jobs,
                                          '[\"output_path\"]': self._model_paths,
                                      })
        self.create_jobs_from_job_config_files(job_config_files=config_files,
                                               job_config_object=job_config_object)

    def prepare_evaluate_model(self, base_config_file: str = None, job_config_object: CondorJobConfig = None,
                               number_of_jobs: int = None):
        base_config = self._config.base_config_files[0] if base_config_file is None else base_config_file
        job_config_object = self._config.job_configs[0] if job_config_object is None else job_config_object
        number_of_jobs = self._config.number_of_jobs[0] if number_of_jobs is None else number_of_jobs
        if number_of_jobs == 0:
            return
        if self._model_paths is None:
            model_directories = [os.path.join(self._config.output_path, 'models', d)
                                 for d in os.listdir(os.path.join(self._config.output_path, 'models'))]
        else:
            model_directories = self._model_paths
        model_directories = model_directories[-min(number_of_jobs, len(model_directories)):]
        with open('src/sim/ros/config/actor/dnn_actor.yml', 'r') as f:
            actor_base_config = yaml.load(f, Loader=yaml.FullLoader)
        actor_base_config['output_path'] = self._config.output_path
        actor_tags = [f'evaluate_{os.path.basename(d)}' for d in model_directories]
        actor_config_files = create_configs(base_config=actor_base_config,
                                            output_path=self._config.output_path,
                                            adjustments={
                                                '[\"specs\"][\"model_config\"][\"load_checkpoint_dir\"]':
                                                    model_directories
                                            })
        config_files = create_configs(base_config=base_config,
                                      output_path=self._config.output_path,
                                      adjustments={
                                          '[\"data_saver_config\"][\"saving_directory_tag\"]': actor_tags,
                                          '[\"runner_config\"][\"environment_config\"]'
                                          '[\"actor_configs\"][0][\"file\"]': actor_config_files
                                      })
        self.create_jobs_from_job_config_files(job_config_files=config_files,
                                               job_config_object=job_config_object)

    def prepare_data_cleaning(self, base_config_file: str = None, job_config_object: CondorJobConfig = None,
                              number_of_jobs: int = None):
        """Launch condor job in virtualenv to clean raw_data in output_path/raw_data and create hdf5 file"""
        base_config = self._config.base_config_files[0] if base_config_file is None else base_config_file
        job_config_object = self._config.job_configs[0] if job_config_object is None else job_config_object
        number_of_jobs = self._config.number_of_jobs[0] if number_of_jobs is None else number_of_jobs
        if number_of_jobs == 0:
            return
        cleaning_config = create_configs(base_config=base_config,
                                         output_path=self._config.output_path,
                                         adjustments={})
        job_config_object.command += f' --config {cleaning_config[0]}'
        condor_job = CondorJob(config=job_config_object)
        condor_job.write_job_file()
        condor_job.write_executable_file()
        self._jobs.append(condor_job)
        time.sleep(1)

    def prepare_dag_data_collection(self):
        self.prepare_data_collection(base_config_file=self._config.base_config_files[0],
                                     job_config_object=self._config.job_configs[0],
                                     number_of_jobs=self._config.number_of_jobs[0])
        self.prepare_data_cleaning(base_config_file=self._config.base_config_files[1],
                                   job_config_object=self._config.job_configs[1],
                                   number_of_jobs=self._config.number_of_jobs[1])
        dag_lines = '# Prepare_dag_data_collection: \n'
        for index, job in enumerate(self._jobs[:-1]):
            dag_lines += f'JOB data_collection_{index} {job.job_file} \n'
        num_collection_jobs = len(self._jobs) - 1
        dag_lines += f'JOB data_cleaning {self._jobs[-1].job_file} \n'
        dag_lines += f'PARENT {" ".join([f"data_collection_{index}" for index in range(num_collection_jobs)])} ' \
                     f'CHILD data_cleaning \n'
        for index in range(num_collection_jobs):
            dag_lines += f'Retry data_collection_{index} 2 \n'
        dag_lines += f'Retry data_cleaning 3 \n'
        self._dag = Dag(lines_dag_file=dag_lines,
                        dag_directory=os.path.join(self._config.output_path, 'dag', get_date_time_tag()))

    def prepare_dag_train_evaluate(self):
        self.prepare_train_model(base_config_file=self._config.base_config_files[0],
                                 job_config_object=self._config.job_configs[0],
                                 number_of_jobs=self._config.number_of_jobs[0])
        self.prepare_evaluate_model(base_config_file=self._config.base_config_files[1],
                                    job_config_object=self._config.job_configs[1],
                                    number_of_jobs=self._config.number_of_jobs[1])
        dag_lines = '# prepare_dag_train_evaluate: \n'
        for index, job in enumerate(self._jobs[:self._config.number_of_jobs[0]]):
            dag_lines += f'JOB training_{index} {job.job_file} \n'
        for index, job in enumerate(self._jobs[self._config.number_of_jobs[0]:
                                               self._config.number_of_jobs[0] + self._config.number_of_jobs[1]]):
            dag_lines += f'JOB evaluation_{index} {job.job_file} \n'

        number_of_links = min(self._config.number_of_jobs)
        for index in range(number_of_links):
            dag_lines += f'PARENT training_{index} CHILD evaluation_{index} \n'

        for index, job in enumerate(self._jobs[:self._config.number_of_jobs[0]]):
            dag_lines += f'Retry training_{index} 2 \n'
        for index, job in enumerate(self._jobs[self._config.number_of_jobs[0]:
                                               self._config.number_of_jobs[0] + self._config.number_of_jobs[1]]):
            dag_lines += f'Retry evaluation_{index} 3 \n'

        self._dag = Dag(lines_dag_file=dag_lines,
                        dag_directory=os.path.join(self._config.output_path, 'dag', get_date_time_tag()))

    def prepare_dag_data_collection_train_evaluate(self):
        self.prepare_data_collection(base_config_file=self._config.base_config_files[0],
                                     job_config_object=self._config.job_configs[0],
                                     number_of_jobs=self._config.number_of_jobs[0])
        self.prepare_data_cleaning(base_config_file=self._config.base_config_files[1],
                                   job_config_object=self._config.job_configs[1],
                                   number_of_jobs=self._config.number_of_jobs[1])
        self.prepare_train_model(base_config_file=self._config.base_config_files[2],
                                 job_config_object=self._config.job_configs[2],
                                 number_of_jobs=self._config.number_of_jobs[2])
        self.prepare_evaluate_model(base_config_file=self._config.base_config_files[3],
                                    job_config_object=self._config.job_configs[3],
                                    number_of_jobs=self._config.number_of_jobs[3])
        dag_lines = '# Prepare_dag_data_collection_train_evaluate: \n'
        # Define jobs:
        start_index = 0
        end_index = self._config.number_of_jobs[0]
        for index, job in enumerate(self._jobs[start_index: end_index]):
            dag_lines += f'JOB data_collection_{index} {job.job_file} \n'
        start_index = self._config.number_of_jobs[0]
        end_index = sum(self._config.number_of_jobs[0:2])
        assert end_index - start_index == 1
        for index, job in enumerate(self._jobs[start_index: end_index]):
            dag_lines += f'JOB data_cleaning {job.job_file} \n'
        start_index = sum(self._config.number_of_jobs[0:2])
        end_index = sum(self._config.number_of_jobs[0:3])
        for index, job in enumerate(self._jobs[start_index: end_index]):
            dag_lines += f'JOB training_{index} {job.job_file} \n'
        start_index = sum(self._config.number_of_jobs[0:3])
        end_index = sum(self._config.number_of_jobs[:])
        for index, job in enumerate(self._jobs[start_index: end_index]):
            dag_lines += f'JOB evaluation_{index} {job.job_file} \n'
        # Define links:
        dag_lines += f'PARENT {" ".join([f"data_collection_{i}" for i in range(self._config.number_of_jobs[0])])}' \
                     f' CHILD data_cleaning \n'
        dag_lines += f'PARENT data_cleaning' \
                     f' CHILD {" ".join([f"training_{i}" for i in range(self._config.number_of_jobs[2])])} \n'
        number_of_links = min(self._config.number_of_jobs[2:])
        for index in range(number_of_links):
            dag_lines += f'PARENT training_{index} CHILD evaluation_{index} \n'
        # Define retry numbers
        for index in range(self._config.number_of_jobs[0]):
            dag_lines += f'Retry data_collection_{index} 2 \n'
        dag_lines += f'Retry data_cleaning 3 \n'
        for index, job in enumerate(self._jobs[:self._config.number_of_jobs[2]]):
            dag_lines += f'Retry training_{index} 2 \n'
        for index, job in enumerate(self._jobs[:self._config.number_of_jobs[3]]):
            dag_lines += f'Retry evaluation_{index} 3 \n'
        # Create DAG object
        self._dag = Dag(lines_dag_file=dag_lines,
                        dag_directory=os.path.join(self._config.output_path, 'dag', get_date_time_tag()))


if __name__ == '__main__':
    config_file = Parser().parse_args().config
    launcher_config = CondorLauncherConfig().create(config_file=config_file)
    launcher = CondorLauncher(config=launcher_config)
    print(launcher.launch())
    print('finished')
