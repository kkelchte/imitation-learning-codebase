
from src.core.object_factory import ObjectFactory, ConfigFactory
from src.sim.common.data_types import EnvironmentType
from src.sim.gym.gym_environment import GymEnvironment
from src.sim.gazebo.gazebo_environment import GazeboEnvironment

"""Pick correct environment class according to environment type.

"""


class EnvironmentFactory(ObjectFactory):

    def __init__(self):
        self._class_dict = {
            EnvironmentType.Gym: GymEnvironment,
            EnvironmentType.Gazebo: GazeboEnvironment
        }
        super().__init__(self._class_dict)
