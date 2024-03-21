import logging
from aegis.hermes import hermes


class PickleRecorder:
    def __init__(self, odir):
        self.odir = odir

    def write(self, population):

        stage = hermes.get_stage()
        should_skip = hermes.skip("PICKLE_RATE")
        is_first_stage = stage == 1

        if is_first_stage or not should_skip:
            logging.debug(f"pickle recorded at stage {stage}")
            path = self.odir / str(stage)
            population.save_pickle_to(path)
