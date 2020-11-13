from src.core.object_factory import ObjectFactory
from src.ai.trainer import Trainer
from src.ai.domain_adaptation_trainer import DomainAdaptationTrainer
from src.ai.domain_adaptation_trainer_with_deep_supervision import DeepSupervisedDomainAdaptationTrainer
from src.ai.deep_supervision import DeepSupervision
from src.ai.deep_supervision_with_discriminator import DeepSupervisionWithDiscriminator
from src.ai.deep_supervision_confidence import DeepSupervisionConfidence
from src.ai.vpg import VanillaPolicyGradient
from src.ai.ppo import ProximatePolicyGradient

"""Pick correct environment class according to environment type.

"""


class TrainerFactory(ObjectFactory):

    def __init__(self, *args):
        self._class_dict = {
            "BASE": Trainer,
            "DomainAdaptation": DomainAdaptationTrainer,
            "DA": DomainAdaptationTrainer,
            "DomainAdaptationWithDeepSupervision": DeepSupervisedDomainAdaptationTrainer,
            "DADS": DeepSupervisedDomainAdaptationTrainer,
            "DeepSupervision": DeepSupervision,
            "DS": DeepSupervision,
            "DeepSupervisionWithDiscriminator": DeepSupervisionWithDiscriminator,
            "DSDis": DeepSupervisionWithDiscriminator,
            "DeepSupervisionConfidence": DeepSupervisionConfidence,
            "DSConf": DeepSupervisionConfidence,
            "VPG": VanillaPolicyGradient,
            "VanillaPolicyGradient": VanillaPolicyGradient,
            "ProximatePolicyGradient": ProximatePolicyGradient,
            "PPO": ProximatePolicyGradient,
        }
        super().__init__(self._class_dict)
