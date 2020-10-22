from src.ai.ppo import ProximatePolicyGradient
from src.ai.trainer import Trainer
from src.ai.deep_supervision import DeepSupervision
from src.ai.deep_supervision_with_discriminator import DeepSupervisionWithDiscriminator
from src.ai.deep_supervision_confidence import DeepSupervisionConfidence
from src.ai.vpg import VanillaPolicyGradient
from src.core.object_factory import ObjectFactory

"""Pick correct environment class according to environment type.

"""


class TrainerFactory(ObjectFactory):

    def __init__(self, *args):
        self._class_dict = {
            "BASE": Trainer,
            "DeepSupervision": DeepSupervision,
            "DeepSupervisionWithDiscriminator": DeepSupervisionWithDiscriminator,
            "DeepSupervisionConfidence": DeepSupervisionConfidence,
            "VPG": VanillaPolicyGradient,
            "VanillaPolicyGradient": VanillaPolicyGradient,
            "ProximatePolicyGradient": ProximatePolicyGradient,
            "PPO": ProximatePolicyGradient,
        }
        super().__init__(self._class_dict)
