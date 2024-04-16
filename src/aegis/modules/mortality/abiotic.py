"""Cyclic extrinsic mortality
"""

import math
import logging


class Abiotic:
    """

    VISOR
    Abiotic mortality is an optional source of mortality, useful for modeling death by periodic environmental phenomena such as water availability and temperature.
    It has no effect when ABIOTIC_HAZARD_OFFSET and ABIOTIC_HAZARD_AMPLITUDE are set to 0.
    It is modeled using periodic functions with a period of ABIOTIC_HAZARD_PERIOD, amplitude of ABIOTIC_HAZARD_AMPLITUDE,
    shape of ABIOTIC_HAZARD_SHAPE and constant background mortality of ABIOTIC_HAZARD_OFFSET (negative or positive).
    Negative hazard is clipped to zero.
    Available hazard shapes (waveforms) are flat, sinusoidal, square, triangle, sawtooth, ramp (backward sawtooth) and instant (Dirac comb / impulse train).

    # TODO maybe mention which phenomena could be modeled by which shape
    """

    def __init__(self, ABIOTIC_HAZARD_SHAPE, ABIOTIC_HAZARD_OFFSET, ABIOTIC_HAZARD_AMPLITUDE, ABIOTIC_HAZARD_PERIOD):

        self.ABIOTIC_HAZARD_SHAPE = ABIOTIC_HAZARD_SHAPE
        self.ABIOTIC_HAZARD_OFFSET = ABIOTIC_HAZARD_OFFSET
        self.ABIOTIC_HAZARD_AMPLITUDE = ABIOTIC_HAZARD_AMPLITUDE
        self.ABIOTIC_HAZARD_PERIOD = ABIOTIC_HAZARD_PERIOD

        self.func = {
            "flat": self._flat,
            "sinusoidal": self._sinusoidal,
            "triangle": self._triangle,
            "square": self._square,
            "sawtooth": self._sawtooth,
            "ramp": self._ramp,
            "instant": self._instant,
        }[self.ABIOTIC_HAZARD_SHAPE]

        if self.ABIOTIC_HAZARD_SHAPE == "flat" and self.ABIOTIC_HAZARD_AMPLITUDE > 0 and self.ABIOTIC_HAZARD_OFFSET > 0:
            logging.info(
                """
                Note that under flat cnf.abiotic hazard, amplitude and offset have the same effects;
                the total cnf.abiotic mortality is simply their sum.
                """
            )

    def __call__(self, stage):
        return self.func(stage) + self.ABIOTIC_HAZARD_OFFSET

    def _flat(self, stage):
        return self.ABIOTIC_HAZARD_AMPLITUDE

    def _sinusoidal(self, stage):
        return self.ABIOTIC_HAZARD_AMPLITUDE * math.sin(2 * math.pi * stage / self.ABIOTIC_HAZARD_PERIOD)

    def _triangle(self, stage):
        return self.ABIOTIC_HAZARD_AMPLITUDE * (
            1 - 4 * abs(round(stage / self.ABIOTIC_HAZARD_PERIOD - 0.5) - (stage / self.ABIOTIC_HAZARD_PERIOD - 0.5))
        )

    def _square(self, stage):
        return self.ABIOTIC_HAZARD_AMPLITUDE * (
            1 if (stage % self.ABIOTIC_HAZARD_PERIOD) < (self.ABIOTIC_HAZARD_PERIOD / 2) else -1
        )

    def _sawtooth(self, stage):
        return self.ABIOTIC_HAZARD_AMPLITUDE * (
            2 * (stage / self.ABIOTIC_HAZARD_PERIOD - math.floor(stage / self.ABIOTIC_HAZARD_PERIOD + 0.5))
        )

    def _ramp(self, stage):
        return self.ABIOTIC_HAZARD_AMPLITUDE * (stage % self.ABIOTIC_HAZARD_PERIOD) / self.ABIOTIC_HAZARD_PERIOD

    def _instant(self, stage):
        """Mortality function that every ABIOTIC_HAZARD_PERIOD stages kills ABIOTIC_HAZARD_AMPLITUDE of the total living population; stage 0 is unaffected"""
        if stage == 0 or stage % self.ABIOTIC_HAZARD_PERIOD:
            return 0
        return self.ABIOTIC_HAZARD_AMPLITUDE
