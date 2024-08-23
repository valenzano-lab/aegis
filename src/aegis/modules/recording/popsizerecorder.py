from .recorder import Recorder
from aegis.modules.dataclasses.population import Population


class PopsizeRecorder(Recorder):
    def __init__(self, odir):
        self.odir = odir

    def write(self, population: Population, filename: str):
        popsize = len(population)
        path = self.odir / filename
        with open(path, "a") as file_:
            file_.write(f"{popsize}\n")

    def write_before_reproduction(self, population):
        self.write(population, "popsize_before_reproduction")

    def write_after_reproduction(self, population):
        self.write(population, "popsize_after_reproduction")
