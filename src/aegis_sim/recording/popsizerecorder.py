from .recorder import Recorder
from aegis_sim.dataclasses.population import Population


class PopsizeRecorder(Recorder):
    def __init__(self, odir):
        self.odir = odir
        self._buffers = {}  # filename -> list of lines

    def write(self, popsize: int, filename: str):
        if filename not in self._buffers:
            self._buffers[filename] = []
        self._buffers[filename].append(str(popsize))
        # Flush every 100 entries to avoid unbounded memory growth
        if len(self._buffers[filename]) >= 100:
            self._flush(filename)

    def _flush(self, filename):
        if filename in self._buffers and self._buffers[filename]:
            path = self.odir / filename
            with open(path, "a") as file_:
                file_.write("\n".join(self._buffers[filename]) + "\n")
            self._buffers[filename] = []

    def flush_all(self):
        """Flush all buffered data to disk."""
        for filename in list(self._buffers.keys()):
            self._flush(filename)

    def write_before_reproduction(self, population):
        """
        # OUTPUT SPECIFICATION
        path: /popsize_before_reproduction.csv
        filetype: csv
        description: Number of live individuals before reproduction.
        category: demography
        time granularity: every step
        frequency parameter: N/A
        structure: A vector of integers.
        header: None
        """
        self.write(len(population), "popsize_before_reproduction.csv")

    def write_after_reproduction(self, population):
        """
        # OUTPUT SPECIFICATION
        path: /popsize_after_reproduction.csv
        filetype: csv
        description: Number of live individuals after reproduction.
        category: demography
        time granularity: every step
        frequency parameter: N/A
        structure: A vector of integers.
        header: None
        """
        self.write(len(population), "popsize_after_reproduction.csv")

    def write_egg_num_after_reproduction(self, eggs):
        """
        # OUTPUT SPECIFICATION
        path: /eggnum_after_reproduction.csv
        filetype: csv
        description: Number of eggs which have not yet hatched.
        category: demography
        time granularity: every step
        frequency parameter: N/A
        structure: A vector of integers.
        header: None
        """
        eggnum = len(eggs) if eggs is not None else 0
        self.write(eggnum, "eggnum_after_reproduction.csv")
