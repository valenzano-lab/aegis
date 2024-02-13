import pandas as pd
import pathlib
import logging
import json
import yaml
from aegis.help.config import get_default_parameters
from aegis.modules.population import Population
from aegis.help.config import causeofdeath_valid


class Container:
    """Wrapper class
    Contains paths to output files which it can read and return.
    """

    def __init__(self, basepath):
        self.basepath = pathlib.Path(basepath).absolute()
        self.name = self.basepath.stem
        self.data = {}

        # Set paths
        self.paths = {
            path.stem: path for path in self.basepath.glob("**/*") if path.is_file() and path.suffix == ".csv"
        }
        self.paths["log"] = self.basepath / "progress.log"
        self.paths["output_summary"] = self.basepath / "output_summary.json"
        self.paths["input_summary"] = self.basepath / "input_summary.json"
        self.paths["snapshots"] = {}
        for kind in ("demography", "phenotypes", "genotypes"):
            self.paths["snapshots"][kind] = sorted(
                (self.basepath / "snapshots" / kind).glob("*"),
                key=lambda path: int(path.stem),
            )
        self.paths["pickles"] = sorted(
            (self.basepath / "pickles").glob("*"),
            key=lambda path: int(path.stem),
        )
        self.paths["te"] = sorted(
            (self.basepath / "te").glob("*"),
            key=lambda path: int(path.stem),
        )
        if not self.paths["log"].is_file():
            logging.error(f"No AEGIS log found at path {self.paths['log']}")

    def __str__(self):
        return self.name

    def get_record_structure():
        # TODO
        return

    def report(self):
        """Report present and missing files"""
        # TODO
        return

    ### INTERFACING FUNCTIONS ("API")

    # Tables
    def get_birth_table(self, record_type, record_index=None, normalize=False):
        if record_type == "interval":
            table = self._read_df("age_at_birth")
            if normalize:
                table = table.div(table.sum(1), axis=0)
            table.index.names = ["interval"]
            table.columns.names = ["parental_age"]
            table.columns = table.columns.astype(int)
            return table
        elif record_type == "snapshot":
            raise Exception("Snapshot birth table not available")
        else:
            raise Exception(f"record_type must be interval or snapshot, not {record_type}")

    def get_life_table(self, record_type, record_index=None, normalize=False):
        if record_type == "interval":
            table = self._read_df("additive_age_structure")
            table.index.names = ["interval"]
            table.columns.names = ["age_class"]
            table.columns = table.columns.astype(int)
            return table
        elif record_type == "snapshot":
            MAX_LIFESPAN = self.get_config()["MAX_LIFESPAN"]
            table = (
                self.get_demography(record_type, record_index)
                .ages.value_counts()
                .reindex(range(MAX_LIFESPAN), fill_value=0)
            )
            table.index.names = ["age_class"]
            return table
        else:
            raise Exception(f"record_type must be interval or snapshot, not {record_type}")

    def get_death_table(self, record_type, record_index=None, normalize=False):
        if record_type == "interval":
            table = (
                pd.concat(
                    {causeofdeath: self._read_df(f"age_at_{causeofdeath}") for causeofdeath in causeofdeath_valid}
                )
                .swaplevel()
                .sort_index(level=0)
            )
            table.index.names = ["interval", "cause_of_death"]
            table.columns.names = ["age_class"]
            table.columns = table.columns.astype(int)
            return table
        elif record_type == "snapshot":
            raise Exception("Snapshot death table not available")
        else:
            raise Exception(f"record_type must be interval or snapshot, not {record_type}")

    # Metadata

    def read_metadata(self, which, **kwargs):
        if which == "log":
            return self.get_log(kwargs["reload"])
        elif which == "config":
            return self.get_config()
        elif which == "input_summary":
            return self.get_input_summary()
        elif which == "output_summary":
            return self.get_output_summary()
        else:
            raise Exception(f"{which} is not valid; enter 'log', 'config', 'input_summary' or 'output_summary'")

    def get_log(self, reload=True):
        if ("log" not in self.data) or reload:
            df = pd.read_csv(self.paths["log"], sep="|")
            df.columns = [x.strip() for x in df.columns]

            def dhm_inverse(dhm):
                nums = dhm.replace("`", ":").split(":")
                return int(nums[0]) * 24 * 60 + int(nums[1]) * 60 + int(nums[2])

            # TODO resolve deprecated function
            try:
                df[["ETA", "t1M", "runtime"]].map(dhm_inverse)
            except:
                df[["ETA", "t1M", "runtime"]].applymap(dhm_inverse)
            self.data["log"] = df
        return self.data["log"]

    def get_config(self):
        if "config" not in self.data:
            path = self.basepath.parent / f"{self.basepath.stem}.yml"
            with open(path, "r") as file_:
                custom_config = yaml.safe_load(file_)
            default_config = get_default_parameters()
            self.data["config"] = {**default_config, **custom_config}

        return self.data["config"]

    def get_output_summary(self):
        return self._read_json(self.paths["output_summary"])

    def get_input_summary(self):
        return self._read_json(self.paths["input_summary"])

    # BASIC INPUTS
    # Data
    def get_genotypes(self, record_type, record_index=None, reload=True):
        if record_type == "interval":
            return self._read_df("genotypes", reload=reload)
        elif record_type == "snapshot":
            return self._read_snapshot("genotypes", record_index=record_index)
        else:
            raise Exception(f"record_type must be interval or snapshot, not {record_type}")

    def get_phenotypes(self, record_type, record_index=None, reload=True):
        if record_type == "interval":
            return self._read_df("phenotypes", reload=reload)
        elif record_type == "snapshot":
            return self._read_snapshot("phenotypes", record_index=record_index)
        else:
            raise Exception(f"record_type must be interval or snapshot, not {record_type}")

    def get_demography(self, record_type, record_index=None):
        if record_type == "interval":
            raise Exception("Interval demography not available")
        elif record_type == "snapshot":
            return self._read_snapshot("demography", record_index=record_index)
        else:
            raise Exception(f"record_type must be interval or snapshot, not {record_type}")

    def get_survival_analysis_TE(self, record_index):
        assert record_index < len(self.paths["te"]), "Index out of range"
        data = pd.read_csv(self.paths["te"][record_index])
        data.index.names = ["individual"]
        return data

    ### FORMAT READING FUNCTIONS

    def _read_df(self, stem, reload=True):
        file_read = stem in self.data
        file_exists = stem in self.paths
        # TODO Read also files that are not .csv

        if not file_exists:
            logging.error(f"File {self.paths[stem]} does not exist")
        elif (not file_read) or reload:
            self.data[stem] = pd.read_csv(self.paths[stem])

        return self.data.get(stem, pd.DataFrame())

    # def _return_json(self, stem):
    #     df = self._read_df(stem)
    #     json = df.T.to_json(index=False, orient="split")
    #     return json

    # def _return_json(self):
    #     return json.dumps(self)

    @staticmethod
    def _read_json(path):
        assert path.exists(), f"'{path}' does not exist."
        with open(path, "r") as file_:
            return json.load(file_)

    def _read_snapshot(self, record_type, record_index):
        assert record_type in self.paths["snapshots"], f"No records of '{record_type}' can be found in snapshots"
        assert record_index < len(self.paths["snapshots"][record_type]), "Index out of range"
        return pd.read_feather(self.paths["snapshots"][record_type][record_index])

    def _read_pickle(self, record_index):
        assert record_index < len(self.paths["pickles"]), "Index out of range"
        return Population.load_pickle_from(self.paths["pickles"][record_index])

    ### ANALYSIS FUNCTIONS

    def get_intrinsic_mortality(self):
        phenotypes = self._read_df("phenotypes")
        max_age = self.get_config()["MAX_LIFESPAN"]
        # TODO Ensure that you are slicing the phenotype array at right places
        pdf = phenotypes.iloc[:, :max_age]
        y = 1 - pdf
        return y

    def get_intrinsic_survivorship(self):
        phenotypes = self._read_df("phenotypes")
        max_age = self.get_config()["MAX_LIFESPAN"]
        # TODO Ensure that you are slicing the phenotype array at right places
        pdf = phenotypes.iloc[:, :max_age]
        y = pdf.cumprod(axis=1)
        return y

    def get_intrinsic_fertility(self):
        phenotypes = self._read_df("phenotypes")
        max_age = self.get_config()["MAX_LIFESPAN"]
        # TODO Ensure that you are slicing the phenotype array at right places
        # TODO Ensure that fertility is 0 before maturity
        fertility = phenotypes.iloc[:, max_age:]
        y = fertility
        return y

    # genotypic

    def get_derived_allele_freq(self):
        genotypes = self._read_df("genotypes")
        reference = genotypes.round()
        derived_allele_freq = (
            genotypes.iloc[1:].reset_index(drop=True) - reference.iloc[:-1].reset_index(drop=True)
        ).abs()
        return derived_allele_freq

    def get_sfss(self, bins=10):
        daf = self.get_derived_allele_freq()
        bins = 10
        binsize = 1 / (bins - 1)
        sfss = (daf // binsize).T.apply(lambda l: l.value_counts()).T.fillna(0).astype(int)

        sfss.columns = sfss.columns.astype(int)
        sfss.columns.names = ["frequency_bin"]
        sfss.index.names = ["snapshot"]

        return sfss
