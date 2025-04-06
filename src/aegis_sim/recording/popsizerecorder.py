from .recorder import Recorder
from aegis_sim.dataclasses.population import Population


class PopsizeRecorder(Recorder):
    def __init__(self, odir):
        self.odir = odir

    def write(self, popsize: int, filename: str):
        path = self.odir / filename
        with open(path, "a") as file_:
            file_.write(f"{popsize}\n")

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
