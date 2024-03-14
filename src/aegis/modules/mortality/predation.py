"""
https://bbolker.github.io/math3mb/labNotes/lotkavolterra.pdf
"""

import numpy as np


class Predation:
    def __init__(self, PREDATOR_GROWTH, PREDATION_RATE):
        self.N = 1
        self.PREDATOR_GROWTH = PREDATOR_GROWTH
        self.PREDATION_RATE = PREDATION_RATE

    def __call__(self, K):

        # TODO Add attrition penalty for when there are no living prey around (but there still are eggs)
        # TODO Rename K; it appears it is the prey count
        if K == 0:
            return 0

        # Use Verhulst model
        change = self.N * self.PREDATOR_GROWTH * (1 - self.N / K)
        self.N += change

        ratio = np.log(self.N / K)
        # probability to get killed
        # compute logistically
        # multiplication with 2 so that when n_predators == n_prey, the mortality rate is equal to PREDATION_RATE
        fraction_killed = 2 * self.PREDATION_RATE / (1 + np.exp(-ratio))

        return fraction_killed
