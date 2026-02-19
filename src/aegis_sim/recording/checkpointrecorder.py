"""Recorder that periodically saves full simulation checkpoints."""

import logging
import pathlib

from .recorder import Recorder
from aegis_sim import variables
from aegis_sim.parameterization import parametermanager
from aegis_sim.utilities.funcs import skip


class CheckpointRecorder(Recorder):
    def __init__(self, odir: pathlib.Path):
        self.odir = odir
        self.checkpoint_path = odir / "checkpoint"

    def write(self, population, eggs):
        """Save a checkpoint if CHECKPOINT_RATE says so.

        # OUTPUT SPECIFICATION
        path: /checkpoint
        filetype: pickle
        category: log
        description: Full simulation checkpoint for resuming. Overwritten each time.
        trait granularity: N/A
        time granularity: snapshot
        frequency parameter: CHECKPOINT_RATE
        structure: Binary python file (Checkpoint object).
        """
        from aegis_sim.checkpoint import Checkpoint
        from aegis_sim import submodels
        from aegis_sim.recording import recordingmanager

        if skip("CHECKPOINT_RATE"):
            return

        # Flush buffered recorders so on-disk files are consistent with the checkpoint
        recordingmanager.popsizerecorder.flush_all()
        recordingmanager.resourcerecorder.flush_all()

        step = variables.steps
        checkpoint = Checkpoint.capture(population, eggs, variables, submodels, parametermanager)
        checkpoint.save(self.checkpoint_path)
        logging.debug(f"Checkpoint recorded at step {step}.")
