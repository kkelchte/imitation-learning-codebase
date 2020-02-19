import copy
import os
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import List

import yaml
from dataclasses_json import dataclass_json

from src.condor.condor_job import CondorJobConfig, CondorJob
from src.condor.helper_functions import create_configs
from src.core.config_loader import Parser, Config
from src.core.utils import camelcase_to_snake_format


class CondorLauncherMode(IntEnum):
    DataCollection = 0
    TrainModel = 1
    EvaluateModel = 2


@dataclass_json
@dataclass
class CondorLauncherConfig(Config):
    job_config: CondorJobConfig = None
    mode: CondorLauncherMode = None
    number_of_jobs: int = 1
    use_dag: bool = False
    base_config_file: str = ''


class CondorLauncher:

    def __init__(self, config: CondorLauncherConfig):
        self._config = config
        self._jobs = []
        self.prepare_factory()

    def launch(self):
        if not self._config.use_dag:
            for job in self._jobs:
                job.submit()
        else:
            raise NotImplementedError

    def prepare_factory(self):
        eval(f'self.prepare_{camelcase_to_snake_format(self._config.mode.name)}()')

    def create_jobs_from_job_config_files(self, job_config_files: List[str]) -> None:
        for config in job_config_files:
            job_config = copy.deepcopy(self._config.job_config)
            job_config.config_file = config
            condor_job = CondorJob(config=job_config)
            condor_job.write_job_file()
            condor_job.write_executable_file()
            self._jobs.append(condor_job)
            time.sleep(1)

    def prepare_data_collection(self):
        config_files = create_configs(base_config=self._config.base_config_file,
                                      variable_name='[\"data_saver_config\"][\"saving_directory_tag\"]',
                                      variable_values=list(range(self._config.number_of_jobs)))
        self.create_jobs_from_job_config_files(job_config_files=config_files)

    def prepare_train_model(self):
        config_files = create_configs(base_config=self._config.base_config_file,
                                      variable_name='[\"model_config\"][\"initialisation_seed\"]',
                                      variable_values=[123*n+5100 for n in range(self._config.number_of_jobs)])
        self.create_jobs_from_job_config_files(job_config_files=config_files)

    def prepare_evaluate_model(self):
        model_directories = [os.path.join(self._config.output_path, 'models', d)
                             for d in os.listdir(os.path.join(self._config.output_path, 'models'))]
        model_directories = model_directories[-self._config.number_of_jobs:]
        with open('src/sim/ros/config/actor/dnn_actor.yml', 'r') as f:
            actor_base_config = yaml.load(f, Loader=yaml.FullLoader)
        actor_base_config['output_path'] = self._config.output_path
        actor_config_files = create_configs(base_config=actor_base_config,
                                            variable_name='[\"specs\"][\"model_config\"][\"load_checkpoint_dir\"]',
                                            variable_values=model_directories)
        config_files = create_configs(base_config=self._config.base_config_file,
                                      variable_name='[\"runner_config\"][\"environment_config\"]'
                                                    '[\"actor_configs\"][0][\"file\"]',
                                      variable_values=actor_config_files)
        self.create_jobs_from_job_config_files(job_config_files=config_files)


if __name__ == '__main__':
    config_file = Parser().parse_args().config
    launcher_config = CondorLauncherConfig().create(config_file=config_file)
    launcher = CondorLauncher(config=launcher_config)
    print(launcher.launch())
