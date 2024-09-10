import numpy as np


class Resources:

    def __init__(self, CARRYING_CAPACITY):
        self.CARRYING_CAPACITY = CARRYING_CAPACITY
        self.capacity = 0

    def replenish(self):
        self.capacity = self.CARRYING_CAPACITY

    def reduce(self, amount):
        if amount > self.capacity:
            self.capacity = 0
        else:
            self.capacity -= amount

    def scavenge(self, demands):
        """Return the available amount of resources and reduce exploited amount"""

        total_demand = demands.sum()

        if total_demand > self.capacity:
            ration = self.capacity / len(demands)
            self.reduce(total_demand)
            return ration * np.ones(shape=demands.shape)
        else:
            self.reduce(total_demand)
            return demands
