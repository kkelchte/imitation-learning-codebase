import numpy as np

from src.data.data_saver import DataSaver
from src.core.data_types import Experience, TerminationType


def experience_generator():
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
        experience.observation = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        # experience.action = np.argmax(np.random.multinomial(1, [0.1, 0.8, 0.1]))  # action as unbalanced float
        experience.action = np.asarray([np.argmax(np.random.multinomial(1, [0.1, 0.8, 0.1])),
                                        np.argmax(np.random.multinomial(1, [0.1, 0.8, 0.1])),
                                        0])  # action as array
        experience.reward = np.random.normal()
        yield experience


def generate_dummy_dataset(data_saver: DataSaver, num_runs: int = 10) -> dict:
    episode_lengths = []
    episode_dirs = []
    for run in range(num_runs):
        episode_length = 0
        if run > 0:
            data_saver.update_saving_directory()
        for count, experience in enumerate(experience_generator()):
            if experience.done != TerminationType.Unknown:
                episode_length += 1
            data_saver.save(experience=experience)
        episode_lengths.append(episode_length)
        episode_dirs.append(data_saver.get_saving_directory())
    return {
        'episode_lengths': episode_lengths,
        'episode_directories': episode_dirs
    }
