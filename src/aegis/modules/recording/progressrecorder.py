import logging
import time
import numpy as np

from aegis.hermes import hermes


class ProgressRecorder:
    def __init__(self, odir):
        self.odir = odir
        self.time_start = time.time()

    def write(self, popsize="?"):
        """Record some information about the time and speed of simulation."""

        if hermes.skip("LOGGING_RATE"):
            return

        stage = hermes.get_stage()

        logging.info("%8s / %s / N=%s", stage, hermes.parameters.STAGES_PER_SIMULATION, popsize)

        # Get time estimations
        time_diff = time.time() - self.time_start

        seconds_per_100 = time_diff / stage * 100
        eta = (hermes.parameters.STAGES_PER_SIMULATION - stage) / 100 * seconds_per_100

        stages_per_min = int(stage / (time_diff / 60))

        runtime = self.get_dhm(time_diff)
        time_per_1M = self.get_dhm(time_diff / stage * 1000000)
        eta = self.get_dhm(eta)

        # Save time estimations
        content = (stage, eta, time_per_1M, runtime, stages_per_min, popsize)
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
