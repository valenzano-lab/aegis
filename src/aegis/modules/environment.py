import math


class Environment:
    """Cyclic extrinsic mortality"""

    def __init__(
        self,
        ENVIRONMENT_HAZARD_AMPLITUDE,
        ENVIRONMENT_HAZARD_OFFSET,
        ENVIRONMENT_HAZARD_PERIOD,
        ENVIRONMENT_HAZARD_SHAPE,
    ):
        self.ENVIRONMENT_HAZARD_AMPLITUDE = ENVIRONMENT_HAZARD_AMPLITUDE
        self.ENVIRONMENT_HAZARD_OFFSET = ENVIRONMENT_HAZARD_OFFSET
        self.ENVIRONMENT_HAZARD_PERIOD = ENVIRONMENT_HAZARD_PERIOD
        self.ENVIRONMENT_HAZARD_SHAPE = ENVIRONMENT_HAZARD_SHAPE
        self.func = getattr(self, ENVIRONMENT_HAZARD_SHAPE)

    def __call__(self, stage):
        return self.func(stage) + self.ENVIRONMENT_HAZARD_OFFSET

    def sinusoidal(self, stage):
        return self.ENVIRONMENT_HAZARD_AMPLITUDE * math.sin(
            2 * math.pi * stage / self.ENVIRONMENT_HAZARD_PERIOD
        )

    def triangle(self, stage):
        return self.ENVIRONMENT_HAZARD_AMPLITUDE * (
            1
            - 4
            * abs(
                round(stage / self.ENVIRONMENT_HAZARD_PERIOD - 0.5)
                - (stage / self.ENVIRONMENT_HAZARD_PERIOD - 0.5)
            )
        )

    def square(self, stage):
        return self.ENVIRONMENT_HAZARD_AMPLITUDE * (
            1
            if (stage % self.ENVIRONMENT_HAZARD_PERIOD)
            < (self.ENVIRONMENT_HAZARD_PERIOD / 2)
            else -1
        )

    def sawtooth(self, stage):
        return self.ENVIRONMENT_HAZARD_AMPLITUDE * (
            2
            * (
                stage / self.ENVIRONMENT_HAZARD_PERIOD
                - math.floor(stage / self.ENVIRONMENT_HAZARD_PERIOD + 0.5)
            )
        )

    def ramp(self, stage):
        return (
            self.ENVIRONMENT_HAZARD_AMPLITUDE
            * (stage % self.ENVIRONMENT_HAZARD_PERIOD)
            / self.ENVIRONMENT_HAZARD_PERIOD
        )
