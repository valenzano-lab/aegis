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
        self.write(len(population), "popsize_before_reproduction.csv")

    def write_after_reproduction(self, population):
        self.write(len(population), "popsize_after_reproduction.csv")

    def write_egg_num_after_reproduction(self, eggs):
        eggnum = len(eggs) if eggs is not None else 0
        self.write(eggnum, "eggnum_after_reproduction.csv")
