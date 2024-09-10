import logging
import numpy as np

from aegis_sim.dataclasses.population import Population
from aegis_sim.bioreactor import Bioreactor
from aegis_sim import variables, constants, submodels, parameterization
from aegis_sim.parameterization import parametermanager
from aegis_sim.recording import recordingmanager


def run(custom_config_path, pickle_path, overwrite, custom_input_params):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(module)s: %(message)s",
        datefmt="%d/%m/%Y %I:%M:%S",
        level=logging.DEBUG,
    )
    logging.info(f"Running {custom_config_path}.")

    ##########################
    # INITIALIZATION METHODS #
    ##########################
    variables.steps = 1
    variables.custom_config_path = custom_config_path

    parametermanager.init(
        custom_config_path=custom_config_path,
        custom_input_params=custom_input_params,
    )

    # simname = custom_config_path.stem
    # rng = np.random
    init_traits()
    init_modules()

    variables.random_seed = (
        np.random.randint(1, 10**6)
        if parametermanager.parameters.RANDOM_SEED is None
        else parametermanager.parameters.RANDOM_SEED
    )
    np.random.seed(variables.random_seed)

    recordingmanager.init(custom_config_path, overwrite)
    recordingmanager.initialize_recorders(TICKER_RATE=parametermanager.parameters.TICKER_RATE)
    recordingmanager.configrecorder.write_final_config_file(parametermanager.final_config)

    if pickle_path is None:  # Fresh population
        population = Population.initialize(n=parametermanager.parameters.CARRYING_CAPACITY)
    else:  # Pickled population
        population = Population.load_pickle_from(pickle_path)

    bioreactor = Bioreactor(population)

    ######################
    # SIMULATION METHODS #
    ######################

    # presim
    recordingmanager.ticker.start_process()
    ticker_pid = recordingmanager.ticker.pid
    assert ticker_pid is not None
    recordingmanager.summaryrecorder.write_input_summary(
        ticker_pid=recordingmanager.ticker.pid, pickle_path=pickle_path
    )
    # TODO hacky solution of decrementing and incrementing steps
    variables.steps -= 1
    recordingmanager.featherrecorder.write(bioreactor.population)
    variables.steps += 1
    # sim
    recordingmanager.phenomaprecorder.write()

    not_finished = variables.steps <= parametermanager.parameters.STEPS_PER_SIMULATION
    while not_finished and not recordingmanager.is_extinct():
        recordingmanager.progressrecorder.write(len(bioreactor.population))
        recordingmanager.simpleprogressrecorder.write()
        bioreactor.run_step()
        variables.steps += 1

    # postsim
    recordingmanager.summaryrecorder.write_output_summary()
    logging.info("Simulation finished.")
    recordingmanager.ticker.stop_process()


def init_traits():
    """
    Here the trait order is hardcoded.
    """
    from aegis_sim.parameterization.trait import Trait

    traits = {}
    for traitname in constants.EVOLVABLE_TRAITS:
        trait = Trait(name=traitname, cnf=parametermanager.parameters)
        traits[traitname] = trait

    parameterization.traits = traits


def init_modules():
    # NOTE Circular import if put on top
    # NOTE when modules are put into simplenamespace, no code intelligence can be used
    from aegis_sim.submodels.reproduction.mutation import Mutator
    from aegis_sim.submodels.reproduction.reproduction import Reproducer
    from aegis_sim.submodels.abiotic import Abiotic
    from aegis_sim.submodels.predation import Predation
    from aegis_sim.submodels.resources.starvation import Starvation
    from aegis_sim.submodels.infection import Infection
    from aegis_sim.submodels.frailty import Frailty
    from aegis_sim.submodels.genetics.ploider import Ploider
    from aegis_sim.submodels.genetics.architect import Architect
    from aegis_sim.utilities.popgenstats import PopgenStats
    from aegis_sim.submodels.resources.resources import Resources
    from aegis_sim.submodels.reproduction.sexsystem import SexSystem
    from aegis_sim.submodels.reproduction.matingmanager import MatingManager

    # class Modules:
    #     def __init__(
    #         self,
    #         abiotic=None,
    #         predation=None,
    #         starvation=None,
    #         infection=None,
    #         frailty=None,
    #         resources=None,
    #         mutator=None,
    #         reproduction=None,
    #         matingmanager=None,
    #         ploidy=None,
    #         sexsystem=None,
    #         popgenstats=None,
    #     ):
    #         self.abiotic = abiotic
    #         self.predation = predation
    #         self.starvation = starvation
    #         self.infection = infection
    #         self.frailty = frailty
    #         self.resources = resources
    #         self.mutator = mutator
    #         self.reproduction = reproduction
    #         self.matingmanager = matingmanager
    #         self.ploidy = ploidy
    #         self.sexsystem = sexsystem
    #         self.popgenstats = popgenstats

    # modules = Modules()
    # # Mortality
    submodels.abiotic = Abiotic(
        ABIOTIC_HAZARD_SHAPE=parametermanager.parameters.ABIOTIC_HAZARD_SHAPE,
        ABIOTIC_HAZARD_OFFSET=parametermanager.parameters.ABIOTIC_HAZARD_OFFSET,
        ABIOTIC_HAZARD_AMPLITUDE=parametermanager.parameters.ABIOTIC_HAZARD_AMPLITUDE,
        ABIOTIC_HAZARD_PERIOD=parametermanager.parameters.ABIOTIC_HAZARD_PERIOD,
    )
    submodels.predation = Predation(
        PREDATOR_GROWTH=parametermanager.parameters.PREDATOR_GROWTH,
        PREDATION_RATE=parametermanager.parameters.PREDATION_RATE,
    )
    submodels.starvation = Starvation(
        STARVATION_RESPONSE=parametermanager.parameters.STARVATION_RESPONSE,
        STARVATION_MAGNITUDE=parametermanager.parameters.STARVATION_MAGNITUDE,
        CLIFF_SURVIVORSHIP=parametermanager.parameters.CLIFF_SURVIVORSHIP,
        CARRYING_CAPACITY=parametermanager.parameters.CARRYING_CAPACITY,
    )
    submodels.infection = Infection(
        BACKGROUND_INFECTIVITY=parametermanager.parameters.BACKGROUND_INFECTIVITY,
        TRANSMISSIBILITY=parametermanager.parameters.TRANSMISSIBILITY,
        RECOVERY_RATE=parametermanager.parameters.RECOVERY_RATE,
        FATALITY_RATE=parametermanager.parameters.FATALITY_RATE,
    )
    submodels.frailty = Frailty(
        FRAILTY_MODIFIER=parametermanager.parameters.FRAILTY_MODIFIER,
        AGE_LIMIT=parametermanager.parameters.AGE_LIMIT,
    )

    # Resources
    submodels.resources = Resources(
        CARRYING_CAPACITY=parametermanager.parameters.CARRYING_CAPACITY,
    )

    # Reproduction
    submodels.mutator = Mutator(
        MUTATION_RATIO=parametermanager.parameters.MUTATION_RATIO,
        MUTATION_METHOD=parametermanager.parameters.MUTATION_METHOD,
        MUTATION_AGE_MULTIPLIER=parametermanager.parameters.MUTATION_AGE_MULTIPLIER,
    )
    submodels.reproduction = Reproducer(
        RECOMBINATION_RATE=parametermanager.parameters.RECOMBINATION_RATE,
        REPRODUCTION_MODE=parametermanager.parameters.REPRODUCTION_MODE,
        mutator=submodels.mutator,
    )

    submodels.matingmanager = MatingManager()

    # Genetic architecture
    submodels.ploidy = Ploider(
        REPRODUCTION_MODE=parametermanager.parameters.REPRODUCTION_MODE,
        DOMINANCE_FACTOR=parametermanager.parameters.DOMINANCE_FACTOR,
    )
    submodels.architect = Architect(
        ploid=submodels.ploidy,
        BITS_PER_LOCUS=parametermanager.parameters.BITS_PER_LOCUS,
        PHENOMAP=parametermanager.parameters.PHENOMAP,
        AGE_LIMIT=parametermanager.parameters.AGE_LIMIT,
        THRESHOLD=parametermanager.parameters.THRESHOLD,
        ENVDRIFT_RATE=parametermanager.parameters.ENVDRIFT_RATE,
    )
    submodels.sexsystem = SexSystem()

    # Other
    submodels.popgenstats = PopgenStats()
