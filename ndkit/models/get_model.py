from .registry import MODEL_REGISTRY


def get_model(cfg):
    """
    Build a model instance from the configuration.

    The model class is retrieved from MODEL_REGISTRY, and is expected to follow
    the constructor signature:
        ModelClass(input_size, output_size, cfg)

    Returns:
        nn.Module: Instantiated model.
    """
    model_name = cfg.name

    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model name: '{model_name}'. "
            f"Registered models: {available}"
        )

    ModelClass = MODEL_REGISTRY[model_name]

    return ModelClass(cfg)
