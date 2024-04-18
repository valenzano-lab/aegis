import numpy as np
from aegis.hermes import hermes


class SexSystem:
    def __init__(self):
        pass

    def get_sex(self, n):
        return (hermes.rng.random(n) < 0.5).astype(np.int32)
