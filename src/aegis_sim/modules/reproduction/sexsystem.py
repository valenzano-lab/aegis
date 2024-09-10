import numpy as np


class SexSystem:
    def __init__(self):
        pass

    def get_sex(self, n):
        return (np.random.random(n) < 0.5).astype(np.int32)
