from dataclasses import dataclass
from typing import Optional, List

from dataclasses_json import dataclass_json

from src.core.config_loader import Parser, Config
from src.data.data_saver import DataSaverConfig, DataSaver


@dataclass_json
@dataclass
class DataCleaningConfig(Config):
    data_saver_config: DataSaverConfig = None
    data_directories: Optional[List[str]] = None
    input_size: Optional[List[int]] = None

    def __post_init__(self):
        if self.data_directories is None:
            del self.data_directories
        if self.input_size is None:
            del self.input_size


class DataCleaner:

    def __init__(self, config: DataCleaningConfig):
        self._config = config
        self._data_saver = DataSaver(config=config.data_saver_config)

    def clean(self):
        self._data_saver.create_train_validation_hdf5_files(runs=self._config.data_directories,
                                                            input_size=self._config.input_size)


if __name__ == '__main__':
    config_file = Parser().parse_args().config
    data_cleaning_config = DataCleaningConfig().create(config_file=config_file)
    data_cleaner = DataCleaner(config=data_cleaning_config)
    data_cleaner.clean()
