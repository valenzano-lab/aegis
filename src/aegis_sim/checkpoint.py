"""Checkpoint for resumable simulations.

A Checkpoint captures the full simulation state so it can be restored later,
allowing a simulation to be resumed from exactly where it left off.

How checkpointing works
-----------------------
Checkpoints are saved periodically during a simulation, controlled by the
CHECKPOINT_RATE parameter (in steps). When CHECKPOINT_RATE is 0 (the default),
no checkpoints are saved. The checkpoint is written to ``<output_dir>/checkpoint``
and overwritten each time, so only the latest state is kept on disk.

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

Resuming from a checkpoint
--------------------------
To resume, run::

    python3 -m aegis sim -r <output_directory>

where ``<output_directory>`` is the simulation output folder (e.g.
``temp/test_config``). AEGIS finds the checkpoint file in that directory
and resumes from there.

This is mutually exclusive with ``-c`` (config), ``-o`` (overwrite), and
``-p`` (pickle/seed). On resume, parameters are restored from the saved
config, RNG states are set back, submodels are re-initialized, and all
dynamic state (envdrift map, predator count, resource capacity, eggs) is
injected. Recording continues in append mode — existing output files are
not overwritten and headers are not re-written.

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
        """Serialize checkpoint to disk using atomic write (write to temp, then rename)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with open(fd, "wb") as f:
                pickle.dump(self, f)
            pathlib.Path(tmp_path).replace(path)
        except BaseException:
            pathlib.Path(tmp_path).unlink(missing_ok=True)
            raise
        logging.debug(f"Checkpoint saved at step {self.step} to {path}")

    @classmethod
    def load(cls, path: pathlib.Path) -> "Checkpoint":
        """Deserialize checkpoint from disk."""
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)
        if not isinstance(checkpoint, cls):
            raise TypeError(f"Expected Checkpoint, got {type(checkpoint).__name__}")
        logging.info(f"Checkpoint loaded from {path} (step {checkpoint.step})")
        return checkpoint

    @classmethod
    def find_latest(cls, odir: pathlib.Path) -> pathlib.Path:
        """Find the checkpoint file in an output directory.

        Args:
            odir: The simulation output directory (e.g. ``temp/test_config``).

        Returns:
            Path to the checkpoint file.

        Raises:
            FileNotFoundError: If no checkpoint file is found.
        """
        checkpoint_path = odir / "checkpoint"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint file found in {odir}")
        return checkpoint_path
