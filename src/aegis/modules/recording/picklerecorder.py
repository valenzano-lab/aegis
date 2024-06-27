import logging
import pathlib
from aegis.hermes import hermes

from .recorder import Recorder


class PickleRecorder(Recorder):
    def __init__(self, odir: pathlib.Path):
        self.odir = odir / "pickles"
        self.init_odir()

    def write(self, population):
        """

        # OUTPUT SPECIFICATION
        filetype: pickle
        domain: log
        short description:
        long description:
        content: pickled population
        dtype: (not a matrix)
        index: (not a matrix)
        header: (not a matrix)
        column: (not a matrix)
        rows: (not a matrix)
        path: /pickles/{step}
        """

        step = hermes.get_step()
        should_skip = hermes.skip("PICKLE_RATE")
        is_first_step = step == 1

        if is_first_step or not should_skip:
            logging.debug(f"pickle recorded at step {step}.")
            path = self.odir / str(step)
            population.save_pickle_to(path)
