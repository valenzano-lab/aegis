from .recorder import Recorder
from aegis_sim.submodels.resources.resources import resources


class ResourcesRecorder(Recorder):
    def __init__(self, odir):
        self.odir = odir
        self._buffers = {}

    def _buffered_write(self, value, filename):
        if filename not in self._buffers:
            self._buffers[filename] = []
        self._buffers[filename].append(str(value))
        if len(self._buffers[filename]) >= 100:
            self._flush(filename)

    def _flush(self, filename):
        if filename in self._buffers and self._buffers[filename]:
            path = self.odir / filename
            with open(path, "a") as file_:
                file_.write("\n".join(self._buffers[filename]) + "\n")
            self._buffers[filename] = []

    def flush_all(self):
        for filename in list(self._buffers.keys()):
            self._flush(filename)

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
        self._buffered_write(resources.capacity, "resources_before_scavenging.csv")

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
        self._buffered_write(resources.capacity, "resources_after_scavenging.csv")
