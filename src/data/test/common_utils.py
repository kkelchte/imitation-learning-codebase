import numpy as np

from src.data.data_saver import DataSaver
from src.core.data_types import Experience, TerminationType


def experience_generator(input_size: tuple = (100, 100, 3),
                         output_size: tuple = (1,),
                         continuous: bool = True,
                         fixed_output_value: float = None):
    starting = 5
    running = np.random.randint(10, 12)
    ending = 1
    for step in range(starting + running + ending):
        experience = Experience(info={})
        if step < starting:
            experience.done = TerminationType.Unknown
        elif starting <= step < starting + running:
            experience.done = TerminationType.NotDone
        else:
            experience.done = TerminationType.Success
        experience.time_stamp = step
        experience.observation = np.random.randint(0, 255, size=input_size, dtype=np.uint8)
        if fixed_output_value is not None:
            experience.action = np.asarray(fixed_output_value)
        else:
            if continuous:
                experience.action = np.random.random(output_size)
            else:
                assert len(output_size) == 1
                probabilities = [8]
                probabilities += [1] * (output_size[0] - 1)
                probabilities = [p/sum(probabilities) for p in probabilities]
                experience.action = np.asarray([np.argmax(np.random.multinomial(1, probabilities))])
        experience.reward = np.random.normal()
        yield experience


def generate_dummy_dataset(data_saver: DataSaver,
                           num_runs: int = 10,
                           input_size: tuple = (100, 100, 3),
                           output_size: tuple = (1,),
                           continuous: bool = True,
                           fixed_output_value: float = None,
                           store_hdf5: bool = False) -> dict:
    episode_lengths = []
    episode_dirs = []
    for run in range(num_runs):
        episode_length = 0
        if run > 0:
            data_saver.update_saving_directory()
        for count, experience in enumerate(experience_generator(input_size=input_size,
                                                                output_size=output_size,
                                                                continuous=continuous,
                                                                fixed_output_value=fixed_output_value)):
            if experience.done != TerminationType.Unknown:
                episode_length += 1
            data_saver.save(experience=experience)
        episode_lengths.append(episode_length)
        episode_dirs.append(data_saver.get_saving_directory())
    if store_hdf5:
        data_saver.create_train_validation_hdf5_files()
    return {
        'episode_lengths': episode_lengths,
        'episode_directories': episode_dirs
    }
