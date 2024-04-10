import pandas as pd
import pathlib
import logging
import json
import yaml
from typing import Union

from aegis.modules.initialization.parameterization.default_parameters import get_default_parameters
from aegis.modules.dataclasses.population import Population
from aegis.constants import VALID_CAUSES_OF_DEATH


# TODO for analysis:
# TODO add 0 to survivorship, think about edge cases
# TODO clean up indices, columns and dtypes
# TODO be explicit about aggregation function


class Container:
    """
    Reads and reformats output files so they are available for internal and external use (prepare for export).
    """

    def __init__(self, basepath):
        self.basepath = pathlib.Path(basepath).absolute()
        self.name = self.basepath.stem
        self.data = {}
        self.set_paths()

    def set_paths(self):
        # TODO smarter way of listing paths; you are capturing te files with number keys e.g. '6': ... /te/6.csv; that's silly
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

    def get_record_structure():
        # TODO
        return

    def report(self):
        """Report present and missing files"""
        # TODO
        return

    def export(self):
        """Export all primary data from the container using general formats"""
        # TODO
        return

    ############
    # METADATA #
    ############

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
            if custom_config is None:
                custom_config = {}
            self.data["config"] = {**default_config, **custom_config}

        return self.data["config"]

    def get_output_summary(self) -> Union[dict, None]:
        path = self.paths["output_summary"]
        if path.exists():
            return self._read_json(path)
        return {}

    def get_input_summary(self):
        return self._read_json(self.paths["input_summary"])

    ##########
    # TABLES #
    ##########

    def get_interval_birth_table(self, normalize=False):
        """
        Observed data.
        Number of births (int) per parental age during an interval of length VISOR_RATE.
        columns.name == parental_age (int)
        index.name == interval (int)
        """
        table = self._read_df("age_at_birth")
        if normalize:
            table = table.div(table.sum(1), axis=0)
        table.index.names = ["interval"]
        table.columns.names = ["parental_age"]
        table.columns = table.columns.astype(int)
        return table

    def get_interval_life_table(self, normalize=False):
        """
        Observed data.
        Number of individuals (int) per age class observed during an interval of length VISOR_RATE.
        columns.name == age_class (int)
        index.name == interval (int)
        """
        table = self._read_df("additive_age_structure")
        table.index.names = ["interval"]
        table.columns.names = ["age_class"]
        table.columns = table.columns.astype(int)
        return table

    def get_snapshot_life_table(self, record_index: int, normalize=False):
        """
        Observed data. Series.
        Number of individuals (int) per age class observed at some simulation stage captured by the record of index record_index.
        name == count
        index.name == age_class
        """
        AGE_LIMIT = self.get_config()["AGE_LIMIT"]
        table = self.get_snapshot_demography(record_index).ages.value_counts().reindex(range(AGE_LIMIT), fill_value=0)
        table.index.names = ["age_class"]
        return table

    def get_interval_death_table(self, normalize=False):
        """
        Observed data. Has a MultiIndex.
        Number of deaths (int) per age class observed during an interval of length VISOR_RATE.
        columns.name == age_class (int)
        index.names == ["interval", "cause_of_death"] (int, str)
        """
        # TODO think about position of axes
        table = (
            pd.concat({causeofdeath: self._read_df(f"age_at_{causeofdeath}") for causeofdeath in VALID_CAUSES_OF_DEATH})
            .swaplevel()
            .sort_index(level=0)
        )
        table.index.names = ["interval", "cause_of_death"]
        table.columns.names = ["age_class"]
        table.columns = table.columns.astype(int)
        return table

    ##########
    # BASICS #
    ##########

    # TODO add better column and index names

    def get_snapshot_genotypes(self, record_index):
        """
        columns .. bit index
        index .. individual index
        value .. True or False
        """
        return self._read_snapshot("genotypes", record_index=record_index)

    def get_snapshot_phenotypes(self, record_index):
        """
        columns .. phenotypic trait index
        index .. individual index
        value .. phenotypic trait value
        """
        return self._read_snapshot("phenotypes", record_index=record_index)

    def get_snapshot_demography(self, record_index):
        """
        columns .. ages, births, birthdays
        index .. individual index
        """
        return self._read_snapshot("demography", record_index=record_index)

    def get_interval_genotypes(self, reload=True):
        """
        columns .. bit index
        index .. record index
        value .. mean bit value
        """
        return self._read_df("genotypes", reload=reload)

    def get_interval_phenotypes(self, reload=True):
        """
        columns .. phenotypic trait index
        index .. record index
        value .. median phenotypic trait value
        """
        return self._read_df("phenotypes", reload=reload)

    def get_survival_analysis_TE(self, record_index):
        """
        columns .. T, E
        index .. individual
        value .. age at event, event (1 .. died, 0 .. alive)
        """
        # TODO error with T and E in the record; they are being appended on top
        assert record_index < len(self.paths["te"]), "Index out of range"
        data = pd.read_csv(self.paths["te"][record_index], header=0)
        data.index.names = ["individual"]
        return data

    #############
    # UTILITIES #
    #############

    def _read_df(self, stem, reload=True):
        file_read = stem in self.data
        file_exists = stem in self.paths
        # TODO Read also files that are not .csv

        if not file_exists:
            logging.error(f"File {self.paths[stem]} does not exist")
        elif (not file_read) or reload:
            self.data[stem] = pd.read_csv(self.paths[stem])

        return self.data.get(stem, pd.DataFrame())

    @staticmethod
    def _read_json(path):
        if not path.exists():
            logging.warning(f"'{path}' does not exist.")
            return None
        with open(path, "r") as file_:
            return json.load(file_)

    def _read_snapshot(self, record_type, record_index):
        assert record_type in self.paths["snapshots"], f"No records of '{record_type}' can be found in snapshots"
        assert record_index < len(self.paths["snapshots"][record_type]), "Index out of range"
        return pd.read_feather(self.paths["snapshots"][record_type][record_index])

    def _read_pickle(self, record_index):
        assert record_index < len(self.paths["pickles"]), "Index out of range"
        return Population.load_pickle_from(self.paths["pickles"][record_index])
