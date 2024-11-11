import numpy as np


class Resources:
    """

    GUI
    Resources module computes and keeps track of the amount of available resources.
    Living individuals scavenge resources during each simulation step, with each individual requiring one unit of resources per step to survive.
    If the number of available resource units falls short of the number of individuals, a starvation response is triggered, as defined by the Starvation module.
    In that case, resources are fully depleted.

    While scavenging reduces the available resources, regeneration occurs each step as well.
    Regeneration logic can be customized to operate in an additive or multiplicative manner, or both.
    Additive regeneration increases the resource pool by a constant amount each step, determined by the [[RESOURCE_ADDITIVE_GROWTH]] parameter.
    In contrast, multiplicative regeneration increases the currently available resources by a factor of 1 + [[RESOURCE_MULTIPLICATIVE_GROWTH]].
    Additionally, there may be a cap on the maximum amount of resources that can be accumulated, which can be controlled using the [[RESOURCE_MAXIMUM_AMOUNT]] parameter.

    By default, resource growth is additive and corresponds to the [[RESOURCE_ADDITIVE_GROWTH]].
    Furthermore, there is no maximum limit on resource growth by default.
    """

    def init(
        self,
        RESOURCE_ADDITIVE_GROWTH,
        RESOURCE_MULTIPLICATIVE_GROWTH,
        RESOURCE_MAXIMUM_AMOUNT,
        RESOURCE_INITIAL_AMOUNT,
    ):

        self.capacity = RESOURCE_INITIAL_AMOUNT
        self.RESOURCE_ADDITIVE_GROWTH = RESOURCE_ADDITIVE_GROWTH
        self.RESOURCE_MULTIPLICATIVE_GROWTH = RESOURCE_MULTIPLICATIVE_GROWTH
        self.RESOURCE_MAXIMUM_AMOUNT = RESOURCE_MAXIMUM_AMOUNT if RESOURCE_MAXIMUM_AMOUNT is not None else np.inf

    def replenish(self):
        self.capacity = self.capacity * (1 + self.RESOURCE_MULTIPLICATIVE_GROWTH) + self.RESOURCE_ADDITIVE_GROWTH
        if self.capacity > self.RESOURCE_MAXIMUM_AMOUNT:
            self.capacity = self.RESOURCE_MAXIMUM_AMOUNT

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
