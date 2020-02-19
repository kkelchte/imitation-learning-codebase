import time
from dataclasses import dataclass
from enum import IntEnum

from dataclasses_json import dataclass_json

from src.condor.condor_job import CondorJobConfig, CondorJob
from src.condor.helper_functions import create_configs
from src.core.config_loader import Parser, Config
from src.core.utils import camelcase_to_snake_format


class CondorLauncherMode(IntEnum):
    DataCollection = 0
    TrainModel = 1


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

    def prepare_data_collection(self):
        job_config = self._config.job_config
        config_files = create_configs(base_config=self._config.base_config_file,
                                      variable_name='[\"data_saver_config\"][\"saving_directory_tag\"]',
                                      variable_values=list(range(self._config.number_of_jobs)))
        for config in config_files:
            job_config.config_file = config
            condor_job = CondorJob(config=job_config)
            condor_job.write_job_file()
            condor_job.write_executable_file()
            self._jobs.append(condor_job)
            time.sleep(1)

    def prepare_train_model(self):
        job_config = self._config.job_config
        config_files = create_configs(base_config=self._config.base_config_file,
                                      variable_name='[\"data_saver_config\"][\"saving_directory_tag\"]',
                                      variable_values=list(range(self._config.number_of_jobs)))
        for config in config_files:
            job_config.config_file = config
            condor_job = CondorJob(config=job_config)
            condor_job.write_job_file()
            condor_job.write_executable_file()
            self._jobs.append(condor_job)
            time.sleep(1)


if __name__ == '__main__':
    config_file = Parser().parse_args().config
    launcher_config = CondorLauncherConfig().create(config_file=config_file)
    launcher = CondorLauncher(config=launcher_config)
    print(launcher.launch())
