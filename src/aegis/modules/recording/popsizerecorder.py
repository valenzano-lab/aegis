from .recorder import Recorder


class PopsizeRecorder(Recorder):
    def __init__(self, odir):
        self.odir = odir

    def write(self, population):
        popsize = len(population)
        path = self.odir / "popsize.csv"
        with open(path, "a") as file_:
            file_.write(f"{popsize}\n")
