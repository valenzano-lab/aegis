import numpy as np


class Predation:
    """
    https://bbolker.github.io/math3mb/labNotes/lotkavolterra.pdf
    """

    def __init__(self, PREDATION_RATE, PREDATOR_GROWTH):
        self.N = 1  # can be a real number
        self.PREDATION_RATE = PREDATION_RATE
        self.PREDATOR_GROWTH = PREDATOR_GROWTH

    def __call__(self, K):
        # Use Verhulst model
        change = self.N * self.PREDATOR_GROWTH * (1 - self.N / K)
        self.N += change

        ratio = np.log(self.N / K)
        # probability to get killed
        # compute logistically
        # multiplication with 2 so that when n_predators == n_prey, the mortality rate is equal to PREDATION_RATE
        fraction_killed = 2 * self.PREDATION_RATE / (1 + np.exp(-ratio))

        return fraction_killed
