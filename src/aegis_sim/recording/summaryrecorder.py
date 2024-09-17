import json
import time
import psutil
import numpy as np
import pathlib
import datetime

from aegis_sim.utilities.get_folder_size import get_folder_size_with_du

from aegis_sim import variables
from .recorder import Recorder


class SummaryRecorder(Recorder):
    """

    Records once.
    """

    def __init__(self, odir: pathlib.Path):
        self.odir = odir
        self.init_odir()
        self.time_start = time.time()
        self.extinct = False

        self.memuse = []
        self.pp = psutil.Process()

    def write_output_summary(self):
        """

        # OUTPUT SPECIFICATION
        path: /output_summary.json
        filetype: json
        category: log
        description: A json dictionary documenting a number of simulation traits at the end of the simulation; e.g. extinction status, random seed, time at start, total runtime, median memory use, total storage use.
        trait granularity: N/A
        time granularity: N/A
        frequency parameter: once
        structure: A json dictionary.
        """
        try:
            storage_use = get_folder_size_with_du(self.odir)
        except:
            storage_use = ""

        summary = {
            "extinct": self.extinct,
            "random_seed": variables.random_seed,
            "time_start": self.time_start,
            "runtime": time.time() - self.time_start,
            "memory_use": self.get_median_memuse(),
            "storage_use": storage_use,
        }
        with open(self.odir / "output_summary.json", "w") as f:
            json.dump(summary, f, indent=4)

    def write_input_summary(self, ticker_pid):
        """

        # OUTPUT SPECIFICATION
        path: /input_summary.json
        filetype: json
        category: genotype
        description: A json dictionary documenting a number of simulation traits at the start of the simulation; e.g. random seed, time at start.
        trait granularity:
        time granularity:
        frequency parameter: once
        structure: A json dictionary.
        """
        summary = {
            "random_seed": variables.random_seed,
            "time_start": self.time_start,
            "pickle_path": str(variables.pickle_path),
            "time_start_formatted": datetime.datetime.fromtimestamp(self.time_start).strftime("%Y-%m-%d %H:%M:%S"),
            "pid": self.pp.pid,
            "ticker_pid": ticker_pid,
        }
        with open(self.odir / "input_summary.json", "w") as f:
            json.dump(summary, f, indent=4)

    def record_memuse(self):
        # TODO refine
        memuse_ = self.pp.memory_info()[0] / float(2**20)
        self.memuse.append(memuse_)
        if len(self.memuse) > 1000:
            self.memuse.pop(0)

    def get_median_memuse(self):
        return np.median(self.memuse)
