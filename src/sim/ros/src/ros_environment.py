
from src.sim.common.actors import Actor
from src.sim.common.data_types import Action, State
from src.sim.common.environment import EnvironmentConfig, Environment
from src.sim.ros.src.process_wrappers import XpraWrapper, RosWrapper


class RosEnvironment(Environment):

    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)
        if config.ros_config.headless:
            self._xpra = XpraWrapper()
        self._ros = RosWrapper(config=config.ros_config.ros_launch_config.__dict__)

    def step(self, action: Action) -> State:
        pass

    def reset(self) -> State:
        pass

    def get_actor(self) -> Actor:
        pass
