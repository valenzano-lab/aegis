import logging
import pathlib

from aegis_sim.dataclasses.population import Population
from aegis_sim.bioreactor import Bioreactor
from aegis_sim import variables, submodels, parameterization
from aegis_sim.parameterization import parametermanager
from aegis_sim.recording import recordingmanager


def run(custom_config_path, pickle_path, overwrite, custom_input_params, resume_path=None, extend_steps=None):
    if resume_path is not None:
        odir = pathlib.Path(resume_path)
        checkpoint_file = odir / "checkpoint"

        if not odir.exists():
            # No output dir yet → fresh run
            logging.info(f"No output directory at {odir}, starting fresh run.")
            init(custom_config_path, overwrite=False, pickle_path=pickle_path, custom_input_params=custom_input_params)
            population = (
                Population.initialize(
                    n=parametermanager.parameters.INITIAL_POPULATION_SIZE,
                    AGE_LIMIT=parametermanager.parameters.AGE_LIMIT,
                )
                if pickle_path is None
                else Population.load_pickle_from(pickle_path)
            )
            eggs = None
        elif not checkpoint_file.exists():
            # Output dir exists but no checkpoint → error
            raise FileNotFoundError(
                f"Output directory {odir} exists but contains no checkpoint file. "
                f"Cannot resume. Use -o to overwrite, or delete the directory."
            )
        else:
            # Output dir + checkpoint → resume
            population, eggs = init_resume(resume_path, extend_steps=extend_steps)
    else:
        init(custom_config_path, overwrite, pickle_path, custom_input_params)
        population = (
            Population.initialize(
                n=parametermanager.parameters.INITIAL_POPULATION_SIZE,
                AGE_LIMIT=parametermanager.parameters.AGE_LIMIT,
            )
            if pickle_path is None
            else Population.load_pickle_from(pickle_path)
        )
        eggs = None
        population = _seed_introgression(population)

        if parametermanager.parameters.LINEAGE_TRACING and population.lineage_id is not None:
            recordingmanager.lineagerecorder.write_initial(population.lineage_id)

    bioreactor = Bioreactor(population)
    bioreactor.eggs = eggs
    sim(bioreactor=bioreactor)


def init(custom_config_path, overwrite=False, pickle_path=None, custom_input_params={}):
    """
    When testing aegis, initialize all modules using this function, e.g.

    import aegis_sim
    aegis_sim.init("_.yml")

    And then you can safely import any module.
    """

    custom_config_path = pathlib.Path(custom_config_path)

    parametermanager.init(
        custom_config_path=custom_config_path,
        custom_input_params=custom_input_params,
    )
    variables.init(
        variables,
        custom_config_path=custom_config_path,
        pickle_path=pickle_path,
        RANDOM_SEED=parametermanager.parameters.RANDOM_SEED,
    )
    parameterization.init_traits(parameterization)
    submodels.init(submodels, parametermanager=parametermanager)

    recordingmanager.init(custom_config_path, overwrite)
    recordingmanager.initialize_recorders(TICKER_RATE=parametermanager.parameters.TICKER_RATE)

    if (
        parametermanager.parameters.LINEAGE_TRACING
        and parametermanager.parameters.REPRODUCTION_MODE != "asexual"
    ):
        logging.warning(
            "LINEAGE_TRACING is True but REPRODUCTION_MODE is %r; lineage IDs will be "
            "assigned to the initial population only and NOT propagated to offspring. "
            "Sexual lineage tracing is not yet implemented.",
            parametermanager.parameters.REPRODUCTION_MODE,
        )


def init_resume(resume_path, extend_steps=None):
    """Initialize all modules from the latest checkpoint in the given output directory.

    Args:
        resume_path: Path to the output directory containing the checkpoint.
        extend_steps: If set, override STEPS_PER_SIMULATION to extend the run.
    """
    from aegis_sim.checkpoint import Checkpoint

    odir = pathlib.Path(resume_path)
    checkpoint_path = Checkpoint.find_latest(odir)
    checkpoint = Checkpoint.load(checkpoint_path)

    # Restore parameters from checkpoint config
    parametermanager.init_from_config(checkpoint.final_config, checkpoint.custom_config_path)

    # Apply --extend override if provided
    if extend_steps is not None:
        if extend_steps <= checkpoint.step:
            raise ValueError(
                f"--extend {extend_steps} must be greater than checkpoint step {checkpoint.step}"
            )
        parametermanager.parameters.STEPS_PER_SIMULATION = extend_steps
        parametermanager.final_config["STEPS_PER_SIMULATION"] = extend_steps
        logging.info(f"Extending simulation to {extend_steps} steps (was {checkpoint.final_config['STEPS_PER_SIMULATION']}).")

    # Restore variables (step, RNG state)
    variables.restore_from_checkpoint(variables, checkpoint)

    # Re-init traits and submodels
    parameterization.init_traits(parameterization)
    submodels.init(submodels, parametermanager=parametermanager)

    # Restore envdrift map if it was active
    if checkpoint.envdrift_map is not None:
        submodels.architect.envdrift.map = checkpoint.envdrift_map

    # Restore predator population size
    submodels.predation.N = checkpoint.predator_population_size

    # Restore resource capacity
    from aegis_sim.submodels.resources.resources import resources
    resources.capacity = checkpoint.resource_capacity

    # Init recording in append mode (don't overwrite, don't write headers)
    recordingmanager.init_for_resume(checkpoint.custom_config_path)
    recordingmanager.initialize_recorders(
        TICKER_RATE=parametermanager.parameters.TICKER_RATE,
        resuming=True,
    )

    # Truncate output files to remove data recorded after the checkpoint step
    recordingmanager.truncate_for_resume(checkpoint.step)

    # Update TE recorder's file counter after truncation may have deleted files
    te_dir = recordingmanager.odir / "te"
    if te_dir.exists():
        remaining = list(te_dir.glob("*.csv"))
        recordingmanager.terecorder.TE_number = len(remaining)

    return checkpoint.population, checkpoint.eggs


def _seed_introgression(population: Population) -> Population:
    """If INTROGRESSION_SOURCE and INTROGRESSION_SEEDS are set, load pop A from pickle,
    sample the requested number of individuals, mark their ancestry as True (introgressed),
    set pop B ancestry to False (native), and merge into one population.
    Returns population unchanged if introgression is not configured.
    """
    import numpy as np

    n_seeds = parametermanager.parameters.INTROGRESSION_SEEDS
    source_path = parametermanager.parameters.INTROGRESSION_SOURCE

    if n_seeds == 0 or source_path is None:
        return population

    source_path = pathlib.Path(source_path)
    pop_a = Population.load_pickle_from(source_path)

    if n_seeds > len(pop_a):
        logging.warning(
            f"INTROGRESSION_SEEDS={n_seeds} exceeds source population size {len(pop_a)}; "
            f"clamping to {len(pop_a)}"
        )
        n_seeds = len(pop_a)

    # Sample n_seeds individuals from pop A
    indices = variables.rng.choice(len(pop_a), size=n_seeds, replace=False)
    pop_a *= indices

    genome_shape = pop_a.genomes.array.shape  # (n_seeds, ploidy, n_loci, bpl)

    # Mark all pop A seeds as introgressed
    pop_a.ancestry = np.ones(genome_shape, dtype=np.bool_)

    # Mark all pop B individuals as native
    pop_b_shape = population.genomes.array.shape
    population.ancestry = np.zeros(pop_b_shape, dtype=np.bool_)

    logging.info(
        f"Introgression: seeding {n_seeds} individuals from {source_path} into population of {len(population)}"
    )

    population += pop_a
    return population


def sim(bioreactor):
    # presim
    recordingmanager.configrecorder.write_final_config_file(parametermanager.final_config)
    recordingmanager.ticker.start_process()
    ticker_pid = recordingmanager.ticker.pid
    assert ticker_pid is not None
    recordingmanager.summaryrecorder.write_input_summary(ticker_pid=recordingmanager.ticker.pid)
    # TODO hacky solution of decrementing and incrementing steps
    variables.steps -= 1
    recordingmanager.featherrecorder.write(bioreactor.population)
    variables.steps += 1

    # sim
    recordingmanager.phenomaprecorder.write()

    # Write initial checkpoint before the loop so there's always something to resume from
    if parametermanager.parameters.CHECKPOINT_RATE > 0:
        from aegis_sim.checkpoint import Checkpoint
        initial_cp = Checkpoint.capture(bioreactor.population, bioreactor.eggs, variables, submodels, parametermanager)
        initial_cp.save(recordingmanager.checkpointrecorder.checkpoint_path)
        logging.debug("Initial checkpoint saved before sim loop.")

    while (variables.steps <= parametermanager.parameters.STEPS_PER_SIMULATION) and not recordingmanager.is_extinct():
        recordingmanager.progressrecorder.write(len(bioreactor.population))
        recordingmanager.simpleprogressrecorder.write()
        bioreactor.run_step()
        variables.steps += 1

    # postsim
    recordingmanager.popsizerecorder.flush_all()
    recordingmanager.resourcerecorder.flush_all()
    recordingmanager.summaryrecorder.write_output_summary()
    logging.info("Simulation finished.")
    recordingmanager.ticker.stop_process()
