import pathlib
import pandas as pd
import json
import yaml


class Output:
    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.experiment_name = self.path.name

        ascending = lambda s: int(s.stem)

        ### SAVE FILE PATHS

        # visor
        self.visor_paths = list((self.path / "0/visor").glob("*.csv"))
        self.visor_spectra_paths = list((self.path / "0/visor/spectra").glob("*.csv"))

        # popgen
        self.popgen_paths = list((self.path / "0/popgen").glob("*"))

        # pickles
        self.pickle_paths = sorted((self.path / "0/pickles").glob("*"), key=ascending)

        # snapshots
        self.demography_paths = sorted(
            (self.path / "0/snapshots/demography").glob("*"), key=ascending
        )
        self.genotypes_paths = sorted(
            (self.path / "0/snapshots/genotypes").glob("*"), key=ascending
        )
        self.phenotypes_paths = sorted(
            (self.path / "0/snapshots/phenotypes").glob("*"), key=ascending
        )

        # other

        self.config_path = self.path.parent / f"{self.experiment_name}.yml"

        ### READ FILES

        for path in self.visor_paths + self.visor_spectra_paths + self.popgen_paths:
            csv = pd.read_csv(path, header=None)
            attr = path.stem
            setattr(self, attr, csv)

        # other
        with open(self.path / "0/output_summary.json", "r") as file_:
            self.output_summary = json.load(file_)
        self.progresslog = pd.read_csv(self.path / "progress.log", sep="|")

        with open(self.config_path, "r") as file_:
            self.config = yaml.safe_load(file_)
