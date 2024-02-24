"""Cyclic extrinsic mortality"""
import math
import logging
from aegis.pan import cnf

func = None  # defined below


def get_hazard(stage):
    return func(stage) + cnf.ABIOTIC_HAZARD_OFFSET


def flat(_):
    return cnf.ABIOTIC_HAZARD_AMPLITUDE


def sinusoidal(stage):
    return cnf.ABIOTIC_HAZARD_AMPLITUDE * math.sin(2 * math.pi * stage / cnf.ABIOTIC_HAZARD_PERIOD)


def triangle(stage):
    return cnf.ABIOTIC_HAZARD_AMPLITUDE * (
        1 - 4 * abs(round(stage / cnf.ABIOTIC_HAZARD_PERIOD - 0.5) - (stage / cnf.ABIOTIC_HAZARD_PERIOD - 0.5))
    )


def square(stage):
    return cnf.ABIOTIC_HAZARD_AMPLITUDE * (
        1 if (stage % cnf.ABIOTIC_HAZARD_PERIOD) < (cnf.ABIOTIC_HAZARD_PERIOD / 2) else -1
    )


def sawtooth(stage):
    return cnf.ABIOTIC_HAZARD_AMPLITUDE * (
        2 * (stage / cnf.ABIOTIC_HAZARD_PERIOD - math.floor(stage / cnf.ABIOTIC_HAZARD_PERIOD + 0.5))
    )


def ramp(stage):
    return cnf.ABIOTIC_HAZARD_AMPLITUDE * (stage % cnf.ABIOTIC_HAZARD_PERIOD) / cnf.ABIOTIC_HAZARD_PERIOD


def instant(stage):
    """Mortality function that every ABIOTIC_HAZARD_PERIOD stages kills ABIOTIC_HAZARD_AMPLITUDE of the total living population; stage 0 is unaffected"""
    if stage == 0 or stage % cnf.ABIOTIC_HAZARD_PERIOD:
        return 0
    return cnf.ABIOTIC_HAZARD_AMPLITUDE


func = {
    "flat": flat,
    "sinusoidal": sinusoidal,
    "triangle": triangle,
    "square": square,
    "sawtooth": sawtooth,
    "ramp": ramp,
    "instant": instant,
}[cnf.ABIOTIC_HAZARD_SHAPE]

if (
    cnf.ABIOTIC_HAZARD_SHAPE == "flat"
    and cnf.ABIOTIC_HAZARD_AMPLITUDE > 0
    and cnf.ABIOTIC_HAZARD_OFFSET > 0
):
    logging.info(
        """
        Note that under flat cnf.abiotic hazard, amplitude and offset have the same effects;
        the total cnf.abiotic mortality is simply their sum.
        """
    )
