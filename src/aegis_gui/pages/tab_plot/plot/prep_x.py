import numpy as np


def get_none(container, **_):
    return


def get_ages(container, **_):
    return np.arange(1, container.get_config()["AGE_LIMIT"] + 1)


def get_steps(container, **kwargs):
    y = kwargs["y"]
    return np.arange(1, len(y) + 1) * container.get_config()["INTERVAL_RATE"]
