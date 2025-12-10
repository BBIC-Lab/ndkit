DATASET_REGISTRY = {}

def register_dataset(name):
    """
    Decorator for registering dataset classes.
    """

    def decorator(cls):
        if name in DATASET_REGISTRY:
            raise ValueError(
                f"Dataset name '{name}' is already registered by "
                f"{DATASET_REGISTRY[name]}. Duplicate registration attempted by {cls}."
            )
        DATASET_REGISTRY[name] = cls
        return cls

    return decorator
