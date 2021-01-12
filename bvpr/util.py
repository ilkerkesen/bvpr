from copy import deepcopy

__all__ = (
    "process_config",
)


def process_config(config, dataset):
    config = deepcopy(config)
    return config