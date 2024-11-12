from .recorder import Recorder
from aegis_sim.submodels.resources.resources import resources


class ResourcesRecorder(Recorder):
    def __init__(self, odir):
        self.odir = odir

    def write_before_scavenging(self):
        path = self.odir / "resources_before_scavenging.csv"
        with open(path, "a") as file_:
            file_.write(f"{resources.capacity}\n")

    def write_after_scavenging(self):
        path = self.odir / "resources_after_scavenging.csv"
        with open(path, "a") as file_:
            file_.write(f"{resources.capacity}\n")
