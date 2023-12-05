"""
https://bbolker.github.io/math3mb/labNotes/lotkavolterra.pdf
"""
import numpy as np

N = 1
PREDATION_RATE = None
PREDATOR_GROWTH = None


def init(self, PREDATION_RATE, PREDATOR_GROWTH):
    self.PREDATION_RATE = PREDATION_RATE
    self.PREDATOR_GROWTH = PREDATOR_GROWTH


def call(K):
    # Use Verhulst model
    global N
    change = N * PREDATOR_GROWTH * (1 - N / K)
    N += change

    ratio = np.log(N / K)
    # probability to get killed
    # compute logistically
    # multiplication with 2 so that when n_predators == n_prey, the mortality rate is equal to PREDATION_RATE
    fraction_killed = 2 * PREDATION_RATE / (1 + np.exp(-ratio))

    return fraction_killed
