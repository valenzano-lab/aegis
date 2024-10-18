import numpy as np


class Resources:

    def __init__(self, CARRYING_CAPACITY, RESOURCE_ADDITIVE_GROWTH, RESOURCE_MULTIPLICATIVE_GROWTH):

        if RESOURCE_ADDITIVE_GROWTH is None:
            self.replenish_additive = CARRYING_CAPACITY
        else:
            self.replenish_additive = RESOURCE_ADDITIVE_GROWTH

        if RESOURCE_MULTIPLICATIVE_GROWTH is None:
            self.replenish_multiplicative = 0
        else:
            self.replenish_multiplicative = RESOURCE_MULTIPLICATIVE_GROWTH

        self.capacity = 0

    def replenish(self):
        self.capacity = self.capacity * self.replenish_multiplicative + self.replenish_additive

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
