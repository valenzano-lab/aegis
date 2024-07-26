import copy
import numpy as np
from aegis.hermes import hermes
from aegis.constants import VALID_CAUSES_OF_DEATH
import pathlib

from .recorder import Recorder


class FlushRecorder(Recorder):
    """

    Records collections.
    """

    def __init__(self, odir: pathlib.Path):

        self.odir = odir / "gui" / "spectra"
        self.init_odir()

        self._collection = {
            "age_at_birth": [0] * hermes.parameters.AGE_LIMIT,
            "additive_age_structure": [0] * hermes.parameters.AGE_LIMIT,
        }

        self._collection.update(
            {f"age_at_{causeofdeath}": [0] * hermes.parameters.AGE_LIMIT for causeofdeath in VALID_CAUSES_OF_DEATH}
        )

        self.collection = copy.deepcopy(self._collection)

        self.init_headers()

    def collect(self, key, ages):
        """Add data into memory which will be recorded later."""
        self.collection[key] += np.bincount(ages, minlength=hermes.parameters.AGE_LIMIT)

    def flush(self):
        """Record data that has been collected over time."""
        # spectra/*.csv | Age distribution of various subpopulations (e.g. population that died of genetic causes)

        if hermes.skip("INTERVAL_RATE"):
            return

        for key, val in self.collection.items():
            self.write_age_at(filename=key, collected_values=val)
            # with open(self.odir / f"{key}.csv", "ab") as f:
            #     array = np.array(val)
            #     np.savetxt(f, [array], delimiter=",", fmt="%i")

        # Reinitialize the collection
        self.collection = copy.deepcopy(self._collection)

    def write_age_at(self, filename, collected_values):
        """

        # OUTPUT SPECIFICATION
        path: /gui/spectra/age_at_{cause}.csv
        filetype: csv
        keywords: demography
        description: Total number of deaths by age and cause of death, within a simulation interval.
        structure: An int matrix.
        """
        with open(self.odir / f"{filename}.csv", "ab") as f:
            array = np.array(collected_values)
            np.savetxt(f, [array], delimiter=",", fmt="%i")

    def init_headers(self):
        for key in self._collection.keys():
            with open(self.odir / f"{key}.csv", "ab") as f:
                array = np.arange(hermes.parameters.AGE_LIMIT)
                np.savetxt(f, [array], delimiter=",", fmt="%i")
