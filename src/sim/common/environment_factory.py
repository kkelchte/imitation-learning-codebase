from src.core.object_factory import ObjectFactory
from src.sim.gym.gym_environment import GymEnvironment
try:
    from src.sim.ros.src.ros_environment import RosEnvironment
except ModuleNotFoundError:
    from mock import Mock
    RosEnvironment = Mock

"""Pick correct environment class according to environment type.

"""


class EnvironmentFactory(ObjectFactory):

    def __init__(self):
        self._class_dict = {
            "GYM": GymEnvironment,
            "ROS": RosEnvironment
        }
        super().__init__(self._class_dict)
