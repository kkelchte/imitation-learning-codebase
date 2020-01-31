import numpy as np

from src.sim.common.data_types import State, TerminalType, Action, ActorType


def state_generator():
    starting = 5
    running = 10
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
        state.sensor_data = {'rgb': np.ones((300, 300, 3)), 'depth': np.ones((360,))}
        state.actor_data = {'expert': Action(actor_name='expert',
                                             actor_type=ActorType.Expert,
                                             value=np.ones((6,)))}
        yield state
