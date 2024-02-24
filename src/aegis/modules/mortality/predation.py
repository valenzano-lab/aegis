"""
https://bbolker.github.io/math3mb/labNotes/lotkavolterra.pdf
"""
import numpy as np
from aegis.pan import cnf

N = 1


def call(K):

    # TODO Add attrition penalty for when there are no living prey around (but there still are eggs)
    # TODO Rename K; it appears it is the prey count
    if K == 0:
        return 0

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
