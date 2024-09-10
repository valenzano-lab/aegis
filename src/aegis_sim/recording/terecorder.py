import logging
import numpy as np
from .recorder import Recorder
from aegis_sim import variables

from aegis_sim.parameterization import parametermanager
# BUG Header duplication; a copy ends up as first row

class TERecorder(Recorder):
    def __init__(self, odir):
        self.odir = odir / "te"
        self.init_odir()
        self.TE_number = 0

    def record(self, T, e):
        """
        Record deaths.
        T .. time / duration (ages)
        E .. event observed (0/alive or 1/dead)

        ###

        To fit this data using lifelines, use this script as inspiration:
            from lifelines import KaplanMeierFitter
            kmf = KaplanMeierFitter()
            te = pd.read_csv("/path/to/te/1.csv")
            kmf.fit(te["T"], te["E"])
            kmf.survival_function_.plot()

        You can compare this to observed survivorship curves:
            analyzer.get_total_survivorship(container).plot()
        """

        assert e in ("alive", "dead")

        step = variables.steps

        # open new file and add header
        if (step % parametermanager.parameters.TE_RATE) == 0 or step == 1:
            data = [["T", "E"]]
            self.write(data, "%s")

        # record deaths
        elif ((step % parametermanager.parameters.TE_RATE) < parametermanager.parameters.TE_DURATION) and e == "dead":
            E = np.repeat(1, len(T))
            data = np.array([T, E]).T
            self.write(data, "%i")

        # flush
        elif (
            ((step % parametermanager.parameters.TE_RATE) == parametermanager.parameters.TE_DURATION)
            or step == parametermanager.parameters.STEPS_PER_SIMULATION
        ) and e == "alive":
            logging.debug(f"Data for survival analysis (T,E) flushed at step {step}.")
            E = np.repeat(0, len(T))
            data = np.array([T, E]).T
            self.write(data, "%i")
            self.TE_number += 1

    def write(self, data, fmt):
        """

        # OUTPUT SPECIFICATION
        path: /te/{te_number}.csv
        filetype: csv
        keywords: demography
        description: Data for survival analysis; time until event (age at death or current viable age) and the event type (1 if death, 0 if still alive).
        structure: An int matrix.
        """
        path = self.odir / f"{self.TE_number}.csv"
        with open(path, "ab") as file_:
            np.savetxt(file_, data, delimiter=",", fmt=fmt)
