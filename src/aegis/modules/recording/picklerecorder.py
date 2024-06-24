import logging
import pathlib
from aegis.hermes import hermes


class PickleRecorder:
    def __init__(self, odir: pathlib.Path):
        self.odir = odir / "pickles"

    def write(self, population):
        """

        # OUTPUT SPECIFICATION
        format: pickle
        content: pickled population
        dtype: (not a matrix)
        index: (not a matrix)
        header: (not a matrix)
        column: (not a matrix)
        rows: (not a matrix)
        path: /pickles/{stage}
        """

        stage = hermes.get_stage()
        should_skip = hermes.skip("PICKLE_RATE")
        is_first_stage = stage == 1

        if is_first_stage or not should_skip:
            logging.debug(f"pickle recorded at stage {stage}")
            path = self.odir / str(stage)
            population.save_pickle_to(path)
