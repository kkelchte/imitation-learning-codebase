import signal
import sys
import time
import warnings

from src.core.config_loader import Parser
from src.sim.common.data_types import TerminalType
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

    environment = RosEnvironment(config=config)
    state = environment.reset(dont_wait=False)
    while state.terminal != TerminalType.Success and state.terminal != TerminalType.Failure:
        state = environment.step(dont_wait=False)
    environment.remove()


if __name__ == '__main__':
    main()
