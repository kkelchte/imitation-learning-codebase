import shutil

import yaml
from dataclasses import dataclass

from dataclasses_json import dataclass_json

from src.condor.preparation_functions.il_preparation_functions import *  # Do not remove
from src.condor.preparation_functions.rl_parameter_study import *  # Do not remove
from src.condor.preparation_functions.line_world_functions import *  # Do not remove
from src.core.config_loader import Parser, Config
from src.core.utils import get_data_dir


@dataclass_json
@dataclass
class CondorLauncherConfig(Config):
    mode: str = None
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
        if 'dag' in self._config.mode:
            self._dag.submit()
        else:
            for job in self._jobs:
                job.submit()

    def prepare_factory(self):
        if 'dag' not in self._config.mode:
            self._jobs = eval(f'prepare_{self._config.mode}(base_config_file=self._config.base_config_files[0],'
                              f'                            job_config_object=self._config.job_configs[0],'
                              f'                            number_of_jobs=self._config.number_of_jobs[0],'
                              f'                            output_path=self._config.output_path)')
        else:
            self._dag = eval(f'prepare_{self._config.mode}(base_config_files=self._config.base_config_files,'
                             f'                            job_configs=self._config.job_configs,'
                             f'                            number_of_jobs=self._config.number_of_jobs,'
                             f'                            output_path=self._config.output_path)')


if __name__ == '__main__':
    arguments = Parser().parse_args()
    config_file = arguments.config
    if arguments.rm:
        with open(config_file, 'r') as f:
            configuration = yaml.load(f, Loader=yaml.FullLoader)
        if not configuration['output_path'].startswith('/'):
            configuration['output_path'] = os.path.join(get_data_dir(os.environ['HOME']), configuration['output_path'])
        shutil.rmtree(configuration['output_path'], ignore_errors=True)

    launcher_config = CondorLauncherConfig().create(config_file=config_file)
    launcher = CondorLauncher(config=launcher_config)
    print(launcher.launch())
    print('finished')
