from .registry import DATASET_REGISTRY


def get_dataset(cfg):
    """
    Build a dataset instance from the configuration.

    The dataset class is retrieved from DATASET_REGISTRY, and is expected to follow
    the constructor signature:
        DatasetClass(cfg)

    Returns:
        nn.Module: Instantiated dataset.
    """
    dataset_name = cfg.name

    if dataset_name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset name: '{dataset_name}'. "
            f"Registered datasets: {available}"
        )

    DatasetClass = DATASET_REGISTRY[dataset_name]
    train_set = DatasetClass(cfg, split='train')
    val_set = DatasetClass(cfg, split='val')
    test_set = DatasetClass(cfg, split='test')
    return train_set, val_set, test_set