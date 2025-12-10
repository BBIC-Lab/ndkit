MODEL_REGISTRY = {}

def register_model(name):
    """
    Decorator for registering model classes.
    """

    def decorator(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(
                f"Model name '{name}' is already registered by "
                f"{MODEL_REGISTRY[name]}. Duplicate registration attempted by {cls}."
            )
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator
