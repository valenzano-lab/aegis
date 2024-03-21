import subprocess
import json
import time
import psutil
import numpy as np

from aegis.hermes import hermes


class SummaryRecorder:
    """
    
    Records once.
    """

    def __init__(self, odir):
        self.odir = odir
        self.time_start = time.time()
        self.extinct = False

        self.memuse = []
        self.pp = psutil.Process()

    @staticmethod
    def get_folder_size_with_du(folder_path):
        result = subprocess.run(["du", "-sh", folder_path], stdout=subprocess.PIPE, text=True)
        return result.stdout.split()[0]

    def record_output_summary(self):
        try:
            storage_use = self.get_folder_size_with_du(self.odir)
        except:
            storage_use = ""

        summary = {
            "extinct": self.extinct,
            "random_seed": hermes.random_seed,
            "time_start": self.time_start,
            "runtime": time.time() - self.time_start,
            "jupyter_path": str(self.odir.absolute()),
            "memory_use": self.get_median_memuse(),
            "storage_use": storage_use,
        }
        with open(self.odir / "output_summary.json", "w") as f:
            json.dump(summary, f, indent=4)

    def record_input_summary(self):
        summary = {
            # "extinct": extinct,
            "random_seed": hermes.random_seed,
            "time_start": self.time_start,
            # "time_end": time.time(),
            "jupyter_path": str(self.odir.absolute()),
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
