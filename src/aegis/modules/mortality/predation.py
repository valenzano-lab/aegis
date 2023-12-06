"""
https://bbolker.github.io/math3mb/labNotes/lotkavolterra.pdf
"""
import numpy as np
from aegis import cnf

N = 1


def call(K):
    # Use Verhulst model
    global N
    change = N * cnf.PREDATOR_GROWTH * (1 - N / K)
    N += change

    ratio = np.log(N / K)
    # probability to get killed
    # compute logistically
    # multiplication with 2 so that when n_predators == n_prey, the mortality rate is equal to PREDATION_RATE
    fraction_killed = 2 * cnf.PREDATION_RATE / (1 + np.exp(-ratio))

    return fraction_killed
