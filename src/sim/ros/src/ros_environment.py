
from src.sim.common.environment import EnvironmentConfig, Environment
from src.sim.ros.src.process_wrappers import XpraWrapper, RosWrapper


class RosEnvironment(Environment):

    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)
        self._xpra = XpraWrapper()
        self._ros = RosWrapper(config.ros_config)
