"""Checkpoint for resumable simulations.

A Checkpoint captures the full simulation state so it can be restored later,
allowing a simulation to be resumed from exactly where it left off.

How checkpointing works
-----------------------
Checkpoints are saved periodically during a simulation, controlled by the
CHECKPOINT_RATE parameter (in steps). When CHECKPOINT_RATE is 0 (the default),
no checkpoints are saved. The checkpoint is written to ``<output_dir>/checkpoint``
and overwritten each time, so only the latest state is kept on disk.

An initial checkpoint is written before the simulation loop starts, so there
is always something to resume from even if the program crashes on the first
step. Subsequent checkpoints are written at the end of each step where
``step % CHECKPOINT_RATE == 0``, overwriting the initial one.

The save is atomic (write to a temp file, then rename) so a crash mid-write
cannot corrupt the checkpoint.

Each checkpoint captures everything needed to reconstruct the simulation:

- **population**: the full Population instance (genomes, ages, phenotypes, etc.)
- **eggs**: unhatched offspring (Population instance, or None), so in-progress
  incubation is preserved when INCUBATION_PERIOD > 0
- **step**: the simulation step at which the checkpoint was taken
- **rng_state**: the numpy Generator (``variables.rng``) bit-generator state,
  so the resumed simulation produces the same random sequence it would have
  if it had never been interrupted
- **legacy_rng_state**: the legacy ``np.random`` global state (used by some
  submodels like envdrift)
- **random_seed**: the original seed, kept for bookkeeping
- **envdrift_map**: the environmental drift XOR map (or None if envdrift is off)
- **predator_population_size**: the current predator count, so predation
  dynamics continue correctly when PREDATION_RATE > 0
- **resource_capacity**: the current available resource amount, so resource
  dynamics continue correctly
- **final_config**: the fully resolved parameter dict, so the resumed run uses
  the exact same parameters without needing the original config file
- **custom_config_path**: path to the original config file, used to locate the
  output directory on resume

CLI usage
---------
``-c`` is always required. ``-r``, ``-o``, and ``-p`` are mutually exclusive::

    aegis sim -c config.yml              # fresh run
    aegis sim -c config.yml -o           # fresh run, overwrite existing output
    aegis sim -c config.yml -p pickle    # seed run (new sim from saved population)
    aegis sim -c config.yml -r           # resume from checkpoint
    aegis sim -c config.yml -r --extend 1500  # resume and extend to 1500 steps

When ``-r`` is used and no output directory exists yet, AEGIS falls back to a
fresh run. If the output directory exists but contains no checkpoint, an error
is raised.

Output truncation on resume
----------------------------
Between checkpoints, recorders append data to output files. If the sim crashes
at step 783 but the last checkpoint was at step 700, the output files contain
data for steps 700–783 that will be re-recorded on resume. To prevent
duplicates, ``truncate_for_resume(checkpoint_step)`` is called during resume
initialization. It:

- Truncates per-step files (popsize, resources, eggs) to ``step - 1`` lines
- Truncates rate-based files (progress, spectra, genotypes, phenotypes, popgen)
  to the number of recordings that occurred before the checkpoint step, plus
  any header lines
- Deletes TE files whose collection window started at or after the checkpoint
- Handles envdriftmap separately (uses ``step % rate == 0`` without the
  step-1 special case)

Snapshot files (feather format) are named by step number, so re-recording
simply overwrites them. One-time files (config, summary) are also overwritten.

Extending a simulation
-----------------------
``--extend N`` overrides ``STEPS_PER_SIMULATION`` to N after loading the
checkpoint, allowing a completed simulation to continue beyond its original
target. N must be greater than the checkpoint step.

Checkpointing vs seeding
-------------------------
Seeding (``-p``) starts a *new* simulation using a saved Population as the
initial population. It resets the step counter to 1, creates fresh output,
and does not restore RNG state or configuration.

Checkpointing (``-r``) *resumes* an existing simulation from the exact point
it was saved. The step counter, RNG state, configuration, and output directory
are all preserved.
"""

import pickle
import logging
import pathlib
import tempfile
import numpy as np

from aegis_sim.dataclasses.population import Population


class Checkpoint:
    """Immutable snapshot of simulation state at a given step.

    Attributes:
        population: The full Population instance at the time of capture.
        eggs: Unhatched offspring (Population instance), or None.
        step: Simulation step number when the checkpoint was taken.
        rng_state: State dict of ``numpy.random.Generator.bit_generator``.
        random_seed: The original random seed used to initialize the simulation.
        legacy_rng_state: State tuple from ``numpy.random.get_state()``.
        envdrift_map: Boolean ndarray (the XOR map), or None if envdrift is disabled.
        predator_population_size: Current predator count (float), for predation dynamics.
        resource_capacity: Current available resource amount (float).
        final_config: Fully resolved parameter dict (default + species + config + overrides).
        custom_config_path: Path to the original ``.yml`` config file.
    """

    def __init__(
        self,
        population: Population,
        eggs,
        step: int,
        rng_state: dict,
        random_seed: int,
        legacy_rng_state: dict,
        envdrift_map,
        predator_population_size: float,
        resource_capacity: float,
        final_config: dict,
        custom_config_path: pathlib.Path,
    ):
        self.population = population
        self.eggs = eggs
        self.step = step
        self.rng_state = rng_state
        self.random_seed = random_seed
        self.legacy_rng_state = legacy_rng_state
        self.envdrift_map = envdrift_map
        self.predator_population_size = predator_population_size
        self.resource_capacity = resource_capacity
        self.final_config = final_config
        self.custom_config_path = custom_config_path

    @classmethod
    def capture(cls, population, eggs, variables, submodels, parametermanager):
        """Capture current simulation state into a Checkpoint."""
        from aegis_sim.submodels.resources.resources import resources

        return cls(
            population=population,
            eggs=eggs,
            step=variables.steps,
            rng_state=variables.rng.bit_generator.state,
            random_seed=variables.random_seed,
            legacy_rng_state=np.random.get_state(),
            envdrift_map=submodels.architect.envdrift.map,
            predator_population_size=submodels.predation.N,
            resource_capacity=resources.capacity,
            final_config=parametermanager.final_config,
            custom_config_path=variables.custom_config_path,
        )

    def save(self, path: pathlib.Path):
        """Serialize checkpoint to disk using atomic write with backup.

        Keeps the previous checkpoint as ``<path>.bak`` so that a SIGKILL
        during the rename window cannot leave the user with zero valid
        checkpoints.  Only promotes the current checkpoint to backup if
        it can be successfully unpickled — a corrupt file is deleted instead.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        backup_path = path.with_suffix(".bak")
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with open(fd, "wb") as f:
                pickle.dump(self, f)
            # Only back up the current checkpoint if it's valid
            if path.exists():
                if self._is_valid_checkpoint(path):
                    path.replace(backup_path)
                else:
                    logging.warning(f"Existing checkpoint at {path} is corrupt; discarding instead of backing up.")
                    path.unlink()
            pathlib.Path(tmp_path).replace(path)
        except BaseException:
            pathlib.Path(tmp_path).unlink(missing_ok=True)
            raise
        logging.debug(f"Checkpoint saved at step {self.step} to {path}")

    @staticmethod
    def _is_valid_checkpoint(path: pathlib.Path) -> bool:
        """Return True if the file at *path* can be unpickled."""
        try:
            with open(path, "rb") as f:
                pickle.load(f)
            return True
        except (pickle.UnpicklingError, EOFError, Exception):
            return False

    @classmethod
    def load(cls, path: pathlib.Path) -> "Checkpoint":
        """Deserialize checkpoint from disk, falling back to backup if corrupt."""
        backup_path = path.with_suffix(".bak")
        try:
            return cls._load_single(path)
        except (pickle.UnpicklingError, EOFError) as primary_err:
            if backup_path.exists():
                logging.warning(
                    f"Primary checkpoint at {path} is corrupt ({primary_err}); "
                    f"falling back to backup at {backup_path}."
                )
                return cls._load_single(backup_path)
            raise

    @classmethod
    def _load_single(cls, path: pathlib.Path) -> "Checkpoint":
        """Load and validate a single checkpoint file."""
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)
        if not isinstance(checkpoint, cls):
            raise TypeError(f"Expected Checkpoint, got {type(checkpoint).__name__}")
        logging.info(f"Checkpoint loaded from {path} (step {checkpoint.step})")
        return checkpoint

    @classmethod
    def find_latest(cls, odir: pathlib.Path) -> pathlib.Path:
        """Find the checkpoint file in an output directory.

        Returns the primary checkpoint path if it exists. If only the backup
        exists (e.g. after a SIGKILL during save), returns the backup path.

        Args:
            odir: The simulation output directory (e.g. ``temp/test_config``).

        Returns:
            Path to the checkpoint file.

        Raises:
            FileNotFoundError: If no checkpoint file is found.
        """
        checkpoint_path = odir / "checkpoint"
        backup_path = checkpoint_path.with_suffix(".bak")
        if checkpoint_path.exists():
            return checkpoint_path
        if backup_path.exists():
            logging.warning(
                f"No primary checkpoint in {odir}, using backup {backup_path}."
            )
            return backup_path
        raise FileNotFoundError(f"No checkpoint file found in {odir}")
