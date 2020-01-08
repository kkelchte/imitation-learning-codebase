class ObjectFactory:
    def __init__(self, class_dict: dict = None):
        self._builders = {}
        if class_dict is not None:
            for class_key, class_value in class_dict.items():
                self.register_builder(class_key, class_value)

    def register_builder(self, key, builder):
        self._builders[key] = builder

    def create(self, key, **kwargs):
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(**kwargs)