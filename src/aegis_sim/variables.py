import numpy as np


def init(self, custom_config_path, pickle_path, RANDOM_SEED):
    self.steps = 1
    self.custom_config_path = custom_config_path
    self.pickle_path = pickle_path
    self.random_seed = np.random.randint(1, 10**6) if RANDOM_SEED is None else RANDOM_SEED
    np.random.seed(self.random_seed)
