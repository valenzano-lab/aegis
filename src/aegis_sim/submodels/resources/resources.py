import numpy as np


class Resources:
    """

    GUI
    Resources module computes the available resources for the population.
    Living individuals scavenge resources during each simulation step, with each individual requiring one unit of resources per step.
    If the number of available resource units falls short of the number of individuals, a starvation response is triggered, as defined by the Starvation module.
    In that case, resources are fully depleted.

    While scavenging reduces the available resources, regeneration occurs each step as well.
    Regeneration logic can be customized to operate in an additive or multiplicative manner, or both.
    Additive regeneration increases the resource pool by a constant amount each step, determined by the [[RESOURCE_ADDITIVE_GROWTH]] parameter.
    In contrast, multiplicative regeneration increases the currently available resources by a factor of 1 + [[RESOURCE_MULTIPLICATIVE_GROWTH]].
    Additionally, there may be a cap on the maximum amount of resources that can be accumulated, which can be controlled using the [[RESOURCE_MAXIMUM]] parameter.

    By default, resource growth is additive and corresponds to the [[CARRYING_CAPACITY]].
    Furthermore, there is no maximum limit on resource growth by default.
    """

    def init(self, CARRYING_CAPACITY, RESOURCE_ADDITIVE_GROWTH, RESOURCE_MULTIPLICATIVE_GROWTH, RESOURCE_MAXIMUM):

        self.RESOURCE_MAXIMUM = RESOURCE_MAXIMUM if RESOURCE_MAXIMUM is not None else np.inf

        if RESOURCE_ADDITIVE_GROWTH is None:
            self.replenish_additive = CARRYING_CAPACITY
        else:
            self.replenish_additive = RESOURCE_ADDITIVE_GROWTH

        if RESOURCE_MULTIPLICATIVE_GROWTH is None:
            self.replenish_multiplicative = 0
        else:
            self.replenish_multiplicative = RESOURCE_MULTIPLICATIVE_GROWTH

        self.capacity = CARRYING_CAPACITY

    def replenish(self):
        self.capacity = self.capacity * (1 + self.replenish_multiplicative) + self.replenish_additive
        if self.capacity > self.RESOURCE_MAXIMUM:
            self.capacity = self.RESOURCE_MAXIMUM

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


resources = Resources()
