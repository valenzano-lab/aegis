import copy
import numpy as np
from aegis.hermes import hermes
from aegis.constants import VALID_CAUSES_OF_DEATH


class FlushRecorder:
    """
    
    Records collections.
    """
    def __init__(self, odir):

        self.odir = odir

        self._collection = {
            "age_at_birth": [0] * hermes.parameters.MAX_LIFESPAN,
            "additive_age_structure": [0] * hermes.parameters.MAX_LIFESPAN,
        }

        self._collection.update(
            {f"age_at_{causeofdeath}": [0] * hermes.parameters.MAX_LIFESPAN for causeofdeath in VALID_CAUSES_OF_DEATH}
        )

        self.collection = copy.deepcopy(self._collection)

        self.init_headers()

    def collect(self, key, ages):
        """Add data into memory which will be recorded later."""
        self.collection[key] += np.bincount(ages, minlength=hermes.parameters.MAX_LIFESPAN)

    def flush(self):
        """Record data that has been collected over time."""
        # spectra/*.csv | Age distribution of various subpopulations (e.g. population that died of genetic causes)

        if hermes.skip("VISOR_RATE"):
            return

        for key, val in self.collection.items():
            with open(self.odir / f"{key}.csv", "ab") as f:
                array = np.array(val)
                np.savetxt(f, [array], delimiter=",", fmt="%i")

        # Reinitialize the collection
        self.collection = copy.deepcopy(self._collection)

    def init_headers(self):
        for key in self._collection.keys():
            with open(self.odir / f"{key}.csv", "ab") as f:
                array = np.arange(hermes.parameters.MAX_LIFESPAN)
                np.savetxt(f, [array], delimiter=",", fmt="%i")
