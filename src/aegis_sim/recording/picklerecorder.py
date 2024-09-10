import logging
import pathlib

from .recorder import Recorder
from aegis_sim import variables

from aegis_sim.parameterization import parametermanager
from aegis_sim.utilities.funcs import skip

class PickleRecorder(Recorder):
    def __init__(self, odir: pathlib.Path):
        self.odir = odir / "pickles"
        self.init_odir()

    def write(self, population):
        """

        # OUTPUT SPECIFICATION
        path: /pickles/{step}
        filetype: pickle
        keywords: log
        description: A file that records the Population class instance which can be used to seed a future simulation.
        structure: Binary python file.
        """

        step = variables.steps
        should_skip = skip("PICKLE_RATE")
        is_first_step = step == 1
        is_last_step = step == parametermanager.parameters.STEPS_PER_SIMULATION

        if is_first_step or not should_skip or is_last_step:
            logging.debug(f"pickle recorded at step {step}.")
            path = self.odir / str(step)
            population.save_pickle_to(path)
