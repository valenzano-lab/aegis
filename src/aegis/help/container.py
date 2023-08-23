import pandas as pd
import pathlib
import logging


class Container:
    """Wrapper class

    Contains paths to output files which it can read and return.
    """

    def __init__(self, basepath):
        self.basepath = pathlib.Path(basepath).absolute()
        self.paths = {
            path.stem: path
            for path in self.basepath.glob("**/*")
            if path.is_file() and path.suffix == ".csv"
        }
        self.paths["log"] = self.basepath / "progress.log"
        self.data = {}

        if not self.paths["log"].is_file():
            logging.error(f"No AEGIS log found at path {self.paths['log']} ")

    def get_log(self):
        if "log" not in self.data:
            df = pd.read_csv(self.paths["log"], sep="|")
            df.columns = [x.strip() for x in df.columns]

            def dhm_inverse(dhm):
                nums = dhm.replace("`", ":").split(":")
                return int(nums[0]) * 24 * 60 + int(nums[1]) * 60 + int(nums[2])

            df[["ETA", "t1M", "runtime"]].applymap(dhm_inverse)
            self.data["log"] = df
        return self.data["log"]

    def get_df(self, stem):
        file_read = stem in self.data
        file_exists = stem in self.paths
        # TODO Read also files that are not .csv
        if not file_read and file_exists:
            self.data[stem] = pd.read_csv(self.paths[stem])
        return self.data.get(stem, pd.DataFrame())

    def get_json(self, stem):
        df = self.get_df(stem)
        json = df.T.to_json(index=False, orient="split")
        return json
