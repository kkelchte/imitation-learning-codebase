from src.core.object_factory import ObjectFactory
from src.ai.trainer import TrainerType

"""Pick correct environment class according to environment type.

"""


class EnvironmentFactory(ObjectFactory):

    def __init__(self):
        self._class_dict = {

        }
        super().__init__(self._class_dict)
