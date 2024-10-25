import pathlib
import numpy as np

from .recorder import Recorder
from aegis_sim import submodels


class Envdriftmaprecorder(Recorder):
    """

    Records once.
    """

    def __init__(self, odir: pathlib.Path):
        self.odir = odir
        self.init_odir()

    def write(self, step):
        """

        # OUTPUT SPECIFICATION
        path: /envdriftmap.csv
        filetype: csv
        category: genotype
        description: XOR map for genome (0 = original phenotypic effect, 1 = opposite phenotypic effect). Recorded every ENVDRIFT_RATE steps.
        trait granularity: N/A
        time granularity: N/A
        frequency parameter: once
        structure:
        """
        envdrift = submodels.architect.envdrift

        will_evolve = envdrift.will_evolve(step)

        if will_evolve:
            map_ = envdrift.map.flatten()
            with open(self.odir / "envdrift.csv", "ab") as f:
                array = np.array(map_)
                np.savetxt(f, [array], delimiter=",", fmt="%i")
