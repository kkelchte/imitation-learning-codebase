from src.core.config_loader import Config


class ObjectFactory:
    def __init__(self, class_dict: dict = None):
        self._builders = {}
        if class_dict is not None:
            for class_key, class_value in class_dict.items():
                self._register_builder(class_key, class_value)

    def _register_builder(self, key, builder):
        self._builders[key] = builder

    def create(self, config: Config, *args, **kwargs):
        builder = self._builders.get(config.factory_key)
        if not builder:
            raise ValueError(config.factory_key)
        return builder(config, *args, **kwargs)


class ConfigFactory(ObjectFactory):
    """
    Separate factory class for configs in order to use the create function instead of the default init() function.
    BROKEN: create is not called during dataclass-json.from_dict function.
    """

    def __init__(self, class_dict: dict = None):
        super().__init__(class_dict)

    def create(self, config: dict) -> Config:
        builder = self._builders.get(config['factory_key'])
        if isinstance(config, dict):
            return builder().create(config_dict=config)
        elif isinstance(config, str):
            return builder().create(config_file=config)
        else:
            raise IOError('config argument needs to be a dict or a filename.')
