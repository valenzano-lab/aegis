import time
import pathlib

from .recorder import Recorder
from aegis_sim import variables

from aegis_sim.parameterization import parametermanager

class SimpleProgressRecorder(Recorder):
    def __init__(self, odir: pathlib.Path):
        self.odir = odir
        self.init_odir()
        self.time_start = time.time()

        self.progress_path = self.odir / "simpleprogress.log"
        self.write_one(0)
        # self.init_headers()

    # def init_headers(self):
    #     content = ("step", "ETA", "t1M", "runtime", "stg/min", "popsize")
    #     with open(self.odir / "progress.log", "ab") as f:
    #         np.savetxt(f, [content], fmt="%-10s", delimiter="| ")

    def write(self):
        """Record some information about the time and speed of simulation."""

        last_updated = self.check_when_last_updated(file_path=self.progress_path)
        if last_updated > 1:
            step = variables.steps
            self.write_one(step)

    def write_one(self, step):
        message = f"{step}/{parametermanager.parameters.STEPS_PER_SIMULATION}"
        with open(self.progress_path, "w") as file_:
            file_.write(message)

    @staticmethod
    def check_when_last_updated(file_path: pathlib.Path):
        last_modified_time = file_path.stat().st_mtime
        time_since_modification = time.time() - last_modified_time
        seconds = time_since_modification
        return seconds


        # # Get time estimations
        # time_diff = time.time() - self.time_start

        # seconds_per_100 = time_diff / step * 100
        # eta = (parametermanager.parameters.STEPS_PER_SIMULATION - step) / 100 * seconds_per_100

        # steps_per_min = int(step / (time_diff / 60))

        # runtime = self.get_dhm(time_diff)
        # time_per_1M = self.get_dhm(time_diff / step * 1000000)
        # eta = self.get_dhm(eta)

        # # Save time estimations
        # content = (step, eta, time_per_1M, runtime, steps_per_min, popsize)
        # self.write_to_progress_log(content)

    # def write_to_progress_log(self, content):
    #     """

    #     # OUTPUT SPECIFICATION
    #     path: /progress.log
    #     filetype: txt
    #     keywords: log
    #     description: A table documenting the estimated time of simulation completion (ETA), time to run one million steps (t1M), time since simulation start (runtime), number of simulated steps per minute (stg/min) and population size (popsize).
    #     structure: A str table with custom separator (` | `).
    #     """
    #     with open(self.odir / "progress.log", "ab") as f:
    #         np.savetxt(f, [content], fmt="%-10s", delimiter="| ")

    # @staticmethod
    # def get_dhm(timediff):
    #     """Format time in a human-readable format."""
    #     d = int(timediff / 86400)
    #     timediff %= 86400
    #     h = int(timediff / 3600)
    #     timediff %= 3600
    #     m = int(timediff / 60)
    #     return f"{d}`{h:02}:{m:02}"
