import numpy as np


def get_none(container, **_):
    return


def get_ages(container, **_):
    return np.arange(1, container.get_config()["MAX_LIFESPAN"] + 1)


def get_stages(container, **kwargs):
    y = kwargs["y"]
    return np.arange(1, len(y) + 1) * container.get_config()["VISOR_RATE"]
