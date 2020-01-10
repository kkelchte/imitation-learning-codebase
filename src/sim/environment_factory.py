
from src.core.object_factory import ObjectFactory
from src.sim.gym.gym_environment import GymEnvironment
from src.sim.gazebo.gazebo_environment import GazeboEnvironment

"""Pick correct environment class according to environment type name.

"""


class EnvironmentFactory(ObjectFactory):

    def __init__(self):
        self._class_dict = {
            'gym': GymEnvironment,
            'gazebo': GazeboEnvironment
        }
        super().__init__(self._class_dict)
