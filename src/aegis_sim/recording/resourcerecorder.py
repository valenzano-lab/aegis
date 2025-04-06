from .recorder import Recorder
from aegis_sim.submodels.resources.resources import resources


class ResourcesRecorder(Recorder):
    def __init__(self, odir):
        self.odir = odir

    def write_before_scavenging(self):
        """
        # OUTPUT SPECIFICATION
        path: /resources_before_scavenging.csv
        filetype: csv
        description: Amount of available resources before scavenging.
        category: demography
        time granularity: every step
        frequency parameter: N/A
        structure: A vector of numbers.
        header: None
        """
        path = self.odir / "resources_before_scavenging.csv"
        with open(path, "a") as file_:
            file_.write(f"{resources.capacity}\n")

    def write_after_scavenging(self):
        """
        # OUTPUT SPECIFICATION
        path: /resources_after_scavenging.csv
        filetype: csv
        description: Amount of available resources after scavenging.
        category: demography
        time granularity: every step
        frequency parameter: N/A
        structure: A vector of numbers.
        header: None
        """
        path = self.odir / "resources_after_scavenging.csv"
        with open(path, "a") as file_:
            file_.write(f"{resources.capacity}\n")
