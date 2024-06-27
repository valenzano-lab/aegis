import numpy as np


class Predation:
    """
    VISOR
    Predation is an optional source of mortality, useful for modeling death with prey-predator dynamics.
    [[PREDATION_RATE]] specifies how deadly the predators are; thus if set to 0, no predation deaths will occur.
    Apart from [[PREDATION_RATE]], the probability that an individual actually gets predated depends also on the number of predators; the response curve is logistic.
    The predator population grows according to the logistic Verhulst growth model, whose slope is parameterized by [[PREDATOR_GROWTH]].
    All individuals are equally susceptible to predation; age and genetics have no impact.
    """

    def __init__(self, PREDATOR_GROWTH, PREDATION_RATE):
        self.N = 1
        self.PREDATOR_GROWTH = PREDATOR_GROWTH
        self.PREDATION_RATE = PREDATION_RATE

    def __call__(self, prey_count):

        # TODO Add attrition penalty for when there are no living prey around (but there still are eggs)
        if prey_count == 0:
            return 0

        # Use Verhulst model
        change = self.N * self.PREDATOR_GROWTH * (1 - self.N / prey_count)
        self.N += change

        ratio = np.log(self.N / prey_count)
        # probability to get killed
        # compute logistically
        # multiplication with 2 so that when n_predators == n_prey, the mortality rate is equal to PREDATION_RATE
        fraction_killed = 2 * self.PREDATION_RATE / (1 + np.exp(-ratio))

        return fraction_killed
