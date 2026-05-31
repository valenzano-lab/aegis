import logging
import pathlib

import numpy as np

from .recorder import Recorder
from aegis_sim import variables
from aegis_sim.parameterization import parametermanager


HEADER = "step,lineage_id,parent_lineage_id\n"


class LineageRecorder(Recorder):
    """Write a growing per-birth log to /lineage/births.csv.

    One row per individual ever created: (step at which it was born,
    its unique lineage_id, its parent's lineage_id). Initial-population
    individuals are written at step 0 with parent_lineage_id=-1.

    Triggered inline from bioreactor.reproduction() and from the run()
    entry point — NOT from the standard end-of-step recorder cycle, because
    it needs to see new births as they happen, not snapshots of the alive
    population.

    LINEAGE_RATE controls flush cadence to disk (every Nth flush call);
    every birth is still recorded regardless of the rate.
    """

    def __init__(self, odir: pathlib.Path):
        self.odir = odir / "lineage"
        self.init_odir()
        self._file_path = self.odir / "births.csv"
        self._file = None
        self._writes_since_flush = 0

    def _ensure_open(self):
        if self._file is not None:
            return
        # Write mode the first time (fresh run); open in append mode otherwise.
        # We detect a fresh run by checking whether the header is already present.
        mode = "a" if self._file_path.exists() and self._file_path.stat().st_size > 0 else "w"
        self._file = open(self._file_path, mode)
        if mode == "w":
            self._file.write(HEADER)

    def write_initial(self, lineage_ids):
        if parametermanager.parameters.LINEAGE_RATE <= 0:
            return
        if lineage_ids is None or len(lineage_ids) == 0:
            return
        self._ensure_open()
        for lid in lineage_ids:
            self._file.write(f"0,{int(lid)},-1\n")
        self._maybe_flush()
        logging.debug(f"lineage initial: wrote {len(lineage_ids)} rows at step 0.")

    def write_births(self, parent_lineage_ids, child_lineage_ids, step):
        if parametermanager.parameters.LINEAGE_RATE <= 0:
            return
        if child_lineage_ids is None or len(child_lineage_ids) == 0:
            return
        assert len(parent_lineage_ids) == len(child_lineage_ids)
        self._ensure_open()
        parent_arr = np.asarray(parent_lineage_ids, dtype=np.int64)
        child_arr = np.asarray(child_lineage_ids, dtype=np.int64)
        for pid, cid in zip(parent_arr.tolist(), child_arr.tolist()):
            self._file.write(f"{step},{cid},{pid}\n")
        self._maybe_flush()

    def _maybe_flush(self):
        self._writes_since_flush += 1
        rate = parametermanager.parameters.LINEAGE_RATE
        if rate > 0 and self._writes_since_flush >= rate:
            self._file.flush()
            self._writes_since_flush = 0

    def close(self):
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    # Compatibility no-op so the recorder can be safely added to the standard
    # end-of-step recording cycle if anyone wires it in there (currently we
    # don't — see module docstring).
    def write(self, population):
        return
