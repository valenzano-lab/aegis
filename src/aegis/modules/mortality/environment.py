"""Cyclic extrinsic mortality"""
import math
import logging

ENVIRONMENT_HAZARD_AMPLITUDE = None
ENVIRONMENT_HAZARD_OFFSET = None
ENVIRONMENT_HAZARD_PERIOD = None
ENVIRONMENT_HAZARD_SHAPE = None
func = None


def get_hazard(stage):
    return func(stage) + ENVIRONMENT_HAZARD_OFFSET


def flat():
    return ENVIRONMENT_HAZARD_AMPLITUDE


def sinusoidal(stage):
    return ENVIRONMENT_HAZARD_AMPLITUDE * math.sin(2 * math.pi * stage / ENVIRONMENT_HAZARD_PERIOD)


def triangle(stage):
    return ENVIRONMENT_HAZARD_AMPLITUDE * (
        1 - 4 * abs(round(stage / ENVIRONMENT_HAZARD_PERIOD - 0.5) - (stage / ENVIRONMENT_HAZARD_PERIOD - 0.5))
    )


def square(stage):
    return ENVIRONMENT_HAZARD_AMPLITUDE * (
        1 if (stage % ENVIRONMENT_HAZARD_PERIOD) < (ENVIRONMENT_HAZARD_PERIOD / 2) else -1
    )


def sawtooth(stage):
    return ENVIRONMENT_HAZARD_AMPLITUDE * (
        2 * (stage / ENVIRONMENT_HAZARD_PERIOD - math.floor(stage / ENVIRONMENT_HAZARD_PERIOD + 0.5))
    )


def ramp(stage):
    return ENVIRONMENT_HAZARD_AMPLITUDE * (stage % ENVIRONMENT_HAZARD_PERIOD) / ENVIRONMENT_HAZARD_PERIOD


def init(
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

    if ENVIRONMENT_HAZARD_SHAPE == "flat" and ENVIRONMENT_HAZARD_AMPLITUDE > 0 and ENVIRONMENT_HAZARD_OFFSET > 0:
        logging.info(
            """
            Note that under flat environmental hazard, amplitude and offset have the same effects;
            the total environmental mortality is simply their sum.
            """
        )
