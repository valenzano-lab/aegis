import pandas as pd
import pathlib


class Container:
    """Wrapper class

    Contains paths to output files which it can read and return.
    """

    def __init__(self, basepath):
        self.basepath = pathlib.Path(basepath)
        self.paths = {
            path.stem: path
            for path in self.basepath.glob("**/*")
            if path.is_file() and path.suffix == ".csv"
        }
        self.data = {}

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
