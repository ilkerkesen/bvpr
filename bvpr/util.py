from copy import deepcopy

__all__ = (
    "process_config",
)


def process_config(config, dataset):
    config = deepcopy(config)
    config["text_encoder"]["corpus"] = dataset.corpus
    return config