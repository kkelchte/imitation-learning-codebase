
from src.core.object_factory import ObjectFactory
from src.sim.common.data_types import EnvironmentType
from src.sim.gym.gym_environment import GymEnvironment
from src.sim.ros.src.ros_environment import RosEnvironment

"""Pick correct environment class according to environment type.

"""


class EnvironmentFactory(ObjectFactory):

    def __init__(self):
        self._class_dict = {
            EnvironmentType.Gym: GymEnvironment,
            EnvironmentType.Ros: RosEnvironment
        }
        super().__init__(self._class_dict)
