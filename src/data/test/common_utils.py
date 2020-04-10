import numpy as np

from src.data.dataset_saver import DataSaver
from src.sim.common.data_types import State, TerminalType, Action, ActorType


def state_generator(inputs: dict = None, outputs: dict = None):
    starting = 5
    running = np.random.randint(7, 13)
    ending = 1
    for step in range(starting + running + ending):
        state = State()
        if step < starting:
            state.terminal = TerminalType.Unknown
        elif starting < step < starting + running:
            state.terminal = TerminalType.NotDone
        else:
            state.terminal = TerminalType.Success
        state.time_stamp_ms = step
        state.sensor_data = {'camera': np.ones((300, 300, 3)), 'depth': np.ones((360,))} if inputs is None else inputs
        state.actor_data = {'expert': Action(actor_name='expert',
                                             actor_type=ActorType.Expert,
                                             value=np.ones((6,)))} if outputs is None else outputs
        yield state


def generate_dummy_dataset(data_saver: DataSaver, num_runs: int = 2) -> dict:
    inputs = {'camera': np.ones((300, 300, 3)), 'depth': np.ones((360,))}
    outputs = {'expert': Action(actor_name='expert',
                                actor_type=ActorType.Expert,
                                value=np.ones((6,)))}
    episode_lengths = []
    episode_dirs = []
    for run in range(num_runs):
        episode_length = 0
        if run > 0:
            data_saver.update_saving_directory()
        inputs = {'camera': np.random.uniform(low=0, high=1, size=(300, 300, 3)), 'depth': 5 * np.random.rand(360)}
        outputs = {'expert': Action(actor_name='expert',
                                    actor_type=ActorType.Expert,
                                    value=np.random.normal(0, 1, size=(6,)))}

        for state in state_generator(
            inputs=inputs,
            outputs=outputs
        ):
            if state.terminal != TerminalType.Unknown:
                episode_length += 1
            data_saver.save(state=state,
                            action=None)
        episode_lengths.append(episode_length)
        episode_dirs.append(data_saver.get_saving_directory())
    return {
        'episode_lengths': episode_lengths,
        'episode_directories': episode_dirs,
        'inputs': inputs.keys(),
        'outputs': outputs.keys()
    }
