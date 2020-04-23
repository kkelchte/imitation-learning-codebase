import numpy as np

from src.data.dataset_saver import DataSaver
from src.sim.common.data_types import Experience, TerminationType, Action


def experience_generator(observation: np.ndarray = np.ones((300, 300, 3), dtype=np.uint8),
                         action: Action = Action(actor_name='expert', value=100),
                         reward: int = 999):
    starting = 5
    running = np.random.randint(10, 12)
    ending = 1
    for step in range(starting + running + ending):
        experience = Experience(info={})
        if step < starting:
            experience.done = TerminationType.Unknown
        elif starting < step < starting + running:
            experience.done = TerminationType.NotDone
        else:
            experience.done = TerminationType.Success
        experience.time_stamp = step
        experience.observation = observation
        experience.action = action
        experience.reward = reward
        yield experience


def generate_dummy_dataset(data_saver: DataSaver, num_runs: int = 2) -> dict:
    episode_lengths = []
    episode_dirs = []
    for run in range(num_runs):
        episode_length = 0
        if run > 0:
            data_saver.update_saving_directory()
        observation = (np.random.rand(100, 100, 3)*255).astype(np.uint8)
        action = np.random.uniform(-1, 1)
        reward = np.random.uniform(-1, 1)
        for count, experience in enumerate(experience_generator(observation=observation,
                                                                action=action,
                                                                reward=reward)):
            if experience.done != TerminationType.Unknown:
                episode_length += 1
            data_saver.save(experience=experience)
        episode_lengths.append(episode_length)
        episode_dirs.append(data_saver.get_saving_directory())
    return {
        'episode_lengths': episode_lengths,
        'episode_directories': episode_dirs
    }
