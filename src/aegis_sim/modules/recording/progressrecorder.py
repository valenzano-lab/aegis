import logging
import time
import numpy as np
import pathlib

from aegis_sim.hermes import hermes
from .recorder import Recorder


class ProgressRecorder(Recorder):
    def __init__(self, odir: pathlib.Path):
        self.odir = odir
        self.init_odir()
        self.time_start = time.time()
        self.init_headers()

    def init_headers(self):
        content = ("step", "ETA", "t1M", "runtime", "stg/min", "popsize")
        with open(self.odir / "progress.log", "ab") as f:
            np.savetxt(f, [content], fmt="%-10s", delimiter="| ")

    def write(self, popsize="?"):
        """Record some information about the time and speed of simulation."""

        if hermes.skip("LOGGING_RATE"):
            return

        step = hermes.get_step()

        logging.info(
            "%s / %s / N=%s / simname=%s", step, hermes.parameters.STEPS_PER_SIMULATION, popsize, hermes.simname
        )

        # Get time estimations
        time_diff = time.time() - self.time_start

        seconds_per_100 = time_diff / step * 100
        eta = (hermes.parameters.STEPS_PER_SIMULATION - step) / 100 * seconds_per_100

        steps_per_min = int(step / (time_diff / 60))

        runtime = self.get_dhm(time_diff)
        time_per_1M = self.get_dhm(time_diff / step * 1000000)
        eta = self.get_dhm(eta)

        # Save time estimations
        content = (step, eta, time_per_1M, runtime, steps_per_min, popsize)
        self.write_to_progress_log(content)

    def write_to_progress_log(self, content):
        """

        # OUTPUT SPECIFICATION
        path: /progress.log
        filetype: txt
        keywords: log
        description: A table documenting the estimated time of simulation completion (ETA), time to run one million steps (t1M), time since simulation start (runtime), number of simulated steps per minute (stg/min) and population size (popsize).
        structure: A str table with custom separator (` | `).
        """
        with open(self.odir / "progress.log", "ab") as f:
            np.savetxt(f, [content], fmt="%-10s", delimiter="| ")

    @staticmethod
    def get_dhm(timediff):
        """Format time in a human-readable format."""
        d = int(timediff / 86400)
        timediff %= 86400
        h = int(timediff / 3600)
        timediff %= 3600
        m = int(timediff / 60)
        return f"{d}`{h:02}:{m:02}"
