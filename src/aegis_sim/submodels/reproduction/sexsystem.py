import numpy as np
from aegis_sim import variables


class SexSystem:
    def __init__(self):
        pass

    def get_sex(self, n):
        return (variables.rng.random(n) < 0.5).astype(np.int32)
