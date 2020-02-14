import os

from src.core.config_loader import Parser
from src.data.dataset_saver import DataSaverConfig, DataSaver

if __name__ == '__main__':
    data_directory = '/esat/opal/kkelchte/experimental_data'
    dataset_name = Parser().parse_args().config
    config_dict = {
        'output_path': os.path.join(data_directory, dataset_name),
        'training_validation_split': 0.9,
        'store_hdf5': True
    }
    config = DataSaverConfig().create(config_dict=config_dict)
    data_saver = DataSaver(config=config)
    data_saver.create_train_validation_hdf5_files()
