import logging
import time
import numpy as np

from aegis.hermes import hermes


class ProgressRecorder:
    def __init__(self, odir):
        self.odir = odir
        self.time_start = time.time()
        self.init_headers()

    def init_headers(self):
        content = ("stage", "ETA", "t1M", "runtime", "stg/min", "popsize")
        with open(self.odir / "progress.log", "ab") as f:
            np.savetxt(f, [content], fmt="%-10s", delimiter="| ")

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
        self.write_to_progress_log(content)

    def write_to_progress_log(self, content):
        """

        # OUTPUT SPECIFICATION
        format: txt
        content: stats of simulation progress
        dtype: complex
        index: none
        header: stage, ETA (estimated time until finished), t1M (time to run one million steps), runtime (runtime until the time of recording), stg/min (number of stages simulated per minute), popsize (population size)
        column:
        rows: one record
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
