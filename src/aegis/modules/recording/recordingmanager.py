"""Data recorder

Records data generated by the simulation.

When thinking about recording additional data, consider that there are three recording methods:
    I. Snapshots (record data from the population at a specific stage)
    II. Flushes (collect data over time then flush)
    III. One-time records
    IV. Other: TE records
"""

import pathlib
import shutil

from .terecorder import TERecorder
from .picklerecorder import PickleRecorder
from .popgenstatsrecorder import PopgenStatsRecorder
from .visorrecorder import VisorRecorder
from .flushrecorder import FlushRecorder
from .featherrecorder import FeatherRecorder
from .phenomaprecorder import PhenomapRecorder
from .summaryrecorder import SummaryRecorder
from .progressrecorder import ProgressRecorder


class RecordingManager:
    def __init__(self, custom_config_path, overwrite):
        odir = self.make_odir(custom_config_path=custom_config_path, overwrite=overwrite)
        self.paths = self.get_paths(odir)
        self.make_subfolders(paths=self.paths.values())
        self.initialize_recorders()

    @staticmethod
    def get_paths(odir: pathlib.Path) -> dict:
        return {
            "BASE_DIR": odir,
            "snapshots_genotypes": odir / "snapshots" / "genotypes",
            "snapshots_phenotypes": odir / "snapshots" / "phenotypes",
            "snapshots_demography": odir / "snapshots" / "demography",
            "visor": odir / "visor",
            "visor_spectra": odir / "visor" / "spectra",
            "pickles": odir / "pickles",
            "popgen": odir / "popgen",
            "phenomap": odir,
            "te": odir / "te",
        }

    def initialize_recorders(self):
        self.terecorder = TERecorder(self.paths["te"])
        self.picklerecorder = PickleRecorder(self.paths["pickles"])
        self.popgenstatsrecorder = PopgenStatsRecorder(self.paths["popgen"])
        self.visorrecorder = VisorRecorder(self.paths["visor"])
        self.flushrecorder = FlushRecorder(self.paths["visor_spectra"])
        self.featherrecorder = FeatherRecorder(
            self.paths["snapshots_genotypes"],
            self.paths["snapshots_phenotypes"],
            self.paths["snapshots_demography"],
        )
        self.phenomaprecorder = PhenomapRecorder(self.paths["phenomap"])
        self.summaryrecorder = SummaryRecorder(self.paths["BASE_DIR"])
        self.progressrecorder = ProgressRecorder(self.paths["BASE_DIR"])

    #############
    # UTILITIES #
    #############

    @staticmethod
    def make_odir(custom_config_path, overwrite) -> pathlib.Path:
        output_path = custom_config_path.parent / custom_config_path.stem  # remove .yml
        is_occupied = output_path.exists() and output_path.is_dir()
        if is_occupied:
            if overwrite:
                shutil.rmtree(output_path)
            else:
                raise Exception(f"{output_path} already exists. To overwrite, add flag --overwrite or -o.")
        return output_path

    @staticmethod
    def make_subfolders(paths):
        for path in paths:
            path.mkdir(exist_ok=True, parents=True)