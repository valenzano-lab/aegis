import pathlib
import yaml
from .recorder import Recorder


class ConfigRecorder(Recorder):
    """

    Records once.
    """

    def __init__(self, odir: pathlib.Path):
        self.odir = odir
        self.path = self.odir / "final_config.yml"

    def write_final_config_file(self, final_config):
        with open(self.path, "w") as file_:
            yaml.dump(final_config, file_)
