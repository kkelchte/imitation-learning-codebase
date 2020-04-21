import signal
import sys
import time
import warnings

from src.core.config_loader import Parser
from src.sim.common.data_types import TerminationType
from src.sim.common.environment import EnvironmentConfig
from src.sim.ros.src.ros_environment import RosEnvironment

warnings.filterwarnings("ignore")
environment: RosEnvironment


def signal_handler(signal_number: int, _) -> None:
    global environment
    print(f'[launch_ros] received signal {signal_number}.')
    while 'environment' not in globals():
        time.sleep(0.1)
    environment.remove()
    sys.exit(0)


def main():
    global environment
    config_file = Parser().parse_args().config
    config = EnvironmentConfig().create(config_file=config_file)

    signal.signal(signal.SIGTERM, signal_handler)
    # try:
    environment = RosEnvironment(config=config)
    state = environment.reset()
    while state.terminal != TerminationType.Success and state.terminal != TerminationType.Failure:
        print(f'state: {environment.fsm_state}')
        state = environment.step()
    environment.remove()


if __name__ == '__main__':
    main()
