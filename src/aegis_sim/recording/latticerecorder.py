"""Lattice spatial snapshot recorder.

Writes a per-individual CSV snapshot of the lattice state at the configured
rate. Columns: step, q, r, age, sex, lineage_id, ancestry_fraction.

`runs/lattice_animate.py` consumes the snapshot files in a sim's
`/lattice/` directory and produces a PNG montage (selected steps) or an
animated GIF.

No-op when LATTICE_MODE is False or LATTICE_RECORD_RATE <= 0.
"""

import logging
import pathlib

import numpy as np

from .recorder import Recorder
from aegis_sim import variables
from aegis_sim.parameterization import parametermanager
from aegis_sim.utilities.funcs import skip


HEADER = "step,q,r,age,sex,lineage_id,ancestry_fraction\n"


class LatticeRecorder(Recorder):
    """Per-step CSV snapshots of every individual's lattice position +
    a handful of attributes useful for color-coding the animation."""

    def __init__(self, odir: pathlib.Path):
        self.odir = odir / "lattice"
        self.init_odir()

    def write(self, population):
        """
        # OUTPUT SPECIFICATION
        path: /lattice/step{step}.csv
        filetype: csv
        category: log
        description: Per-individual lattice positions at a single step. One row per living individual: step, q, r, age, sex, lineage_id, ancestry_fraction. Sex and ancestry-related columns are -1 when their respective tracking is disabled.
        trait granularity: individual
        time granularity: snapshot
        frequency parameter: LATTICE_RECORD_RATE
        structure: CSV.
        """
        if not parametermanager.parameters.LATTICE_MODE:
            return
        if parametermanager.parameters.LATTICE_RECORD_RATE <= 0:
            return
        if population.positions is None or len(population) == 0:
            return

        step = variables.steps
        should_skip = skip("LATTICE_RECORD_RATE")
        is_first_step = step == 1
        is_last_step = step == parametermanager.parameters.STEPS_PER_SIMULATION
        if not (is_first_step or not should_skip or is_last_step):
            return

        n = len(population)
        positions = population.positions  # (n, 2)
        ages = population.ages  # (n,)
        # sexes: existing AEGIS conventions vary; we just dump the raw int values
        sexes = population.sexes if population.sexes is not None else np.full(n, -1, dtype=np.int32)
        lineage_id = (
            population.lineage_id if population.lineage_id is not None
            else np.full(n, -1, dtype=np.int64)
        )
        if population.ancestry is not None:
            ancestry = population.ancestry
            # ancestry is bool, shape (n, ploidy, n_loci, bits_per_locus). Per-individual
            # fraction = mean(True bits) across each individual's whole genome.
            frac = ancestry.reshape(n, -1).mean(axis=1)
        else:
            frac = np.full(n, -1.0, dtype=np.float32)

        path = self.odir / f"step{step}.csv"
        with open(path, "w") as fh:
            fh.write(HEADER)
            for i in range(n):
                fh.write(
                    f"{step},{int(positions[i, 0])},{int(positions[i, 1])},"
                    f"{int(ages[i])},{int(sexes[i])},{int(lineage_id[i])},{float(frac[i]):.4f}\n"
                )
        logging.debug(f"lattice snapshot recorded at step {step}.")
