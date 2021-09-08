def validate_keys(params, legal_keys):
    for key in params:
        if key not in legal_keys:
            raise ValueError(f"'{key}' is not a legal parameter")


def validate_values(params):
    # General
    if not isinstance(params["RANDOM_SEED_"], (int, type(None))):
        raise TypeError("RANDOM_SEED_ must be an integer or null")

    # Runtime
    if not isinstance(params["STAGES_PER_SIMULATION_"], int):
        raise TypeError("STAGES_PER_SIMULATION_ must be an integer")
    if params["STAGES_PER_SIMULATION_"] <= 0:
        raise ValueError("STAGES_PER_SIMULATION_ must be greater than 0")

    if not isinstance(params["LOGGING_RATE_"], int):
        raise TypeError("LOGGING_RATE_ must be an integer")

    if params["LOGGING_RATE_"] < 0:
        raise ValueError("LOGGING_RATE_ must be 0 or greater than 0")

    # Recording
    if not isinstance(params["SNAPSHOT_RATE_"], int):
        raise TypeError("SNAPSHOT_RATE_ must be an integer")

    if params["SNAPSHOT_RATE_"] < 0:
        raise ValueError("SNAPSHOT_RATE_ must be 0 or greater than 0")

    if not isinstance(params["VISOR_RATE_"], int):
        raise TypeError("VISOR_RATE_ must be an integer")

    if params["VISOR_RATE_"] < 0:
        raise ValueError("VISOR_RATE_ must be 0 or greater than 0")

    # Multiple ecosystems
    if not isinstance(params["ECOSYSTEM_NUMBER_"], int):
        raise TypeError("ECOSYSTEM_NUMBER_ must be an integer")

    if params["ECOSYSTEM_NUMBER_"] <= 0:
        raise ValueError("ECOSYSTEM_NUMBER_ must be greater than 0")

    # Ecology
    if not isinstance(params["MAX_POPULATION_SIZE"], int):
        raise TypeError("MAX_POPULATION_SIZE must be an integer")

    if params["MAX_POPULATION_SIZE"] <= 0:
        raise ValueError("MAX_POPULATION_SIZE must be greater than 0")

    if not params["OVERSHOOT_EVENT"] in (
        "treadmill_random",
        "treadmill_boomer",
        "treadmill_zoomer",
        "cliff",
        "starvation",
    ):
        raise TypeError("The specified OVERSHOOT_EVENT is not legal")

    if not isinstance(params["CLIFF_SURVIVORSHIP"], (type(None), float)):
        raise TypeError("CLIFF_SURVIVORSHIP must be a float or null")

    if (
        isinstance(params["CLIFF_SURVIVORSHIP"], float)
        and not 0 < params["CLIFF_SURVIVORSHIP"] < 1
    ):
        raise ValueError("CLIFF_SURVIVORSHIP must be a number in interval (0,1)")

    if not isinstance(params["STAGES_PER_SEASON"], int):
        raise TypeError("STAGES_PER_SEASON must be an integer")

    # Genotype
    if not isinstance(params["MAX_LIFESPAN"], int):
        raise TypeError("MAX_LIFESPAN must be an integer")

    if params["MAX_LIFESPAN"] <= 0:
        raise ValueError("MAX_LIFESPAN must be greater than 0")

    if not isinstance(params["BITS_PER_LOCUS"], int):
        raise TypeError("BITS_PER_LOCUS must be an integer")

    if params["MATURATION_AGE"] <= 0:
        raise ValueError("MATURATION_AGE must be greater than 0")

    if not isinstance(params["HEADSUP"], int):
        raise TypeError("HEADSUP must be an integer")

    if not (params["HEADSUP"] >= 1 or params["HEADSUP"] in (-1, 0)):
        raise ValueError("HEADSUP must be greater -1, 0 or an integer greater than 0")

    # Genome structure
    # - validated when instantiating class Trait

    # Reproduction
    if not params["REPRODUCTION_MODE"] in (
        "sexual",
        "asexual",
        "asexual_diploid",
    ):
        raise ValueError("The specified OVERSHOOT_EVENT is not legal")

    if not isinstance(params["RECOMBINATION_RATE"], (int, float)):
        raise TypeError("RECOMBINATION_RATE must be an integer or a float")

    if not 0 <= params["RECOMBINATION_RATE"] <= 1:
        raise ValueError("RECOMBINATION_RATE must be a number in interval [0,1]")

    if (
        params["REPRODUCTION_MODE"] in ("asexual", "asexual_diploid")
        and not params["RECOMBINATION_RATE"] == 0
    ):
        raise ValueError("Recombination rate must be 0 if reproduction mode is asexual")

    # Mutation
    if not isinstance(params["MUTATION_RATIO"], (int, float)):
        raise TypeError("MUTATION_RATIO must be an integer or a float")

    if not 0 <= params["MUTATION_RATIO"] <= 1:
        raise ValueError("MUTATION_RATIO must be a number in the interval [0,1]")

    # Phenomap
    if not isinstance(params["PHENOMAP_SPECS"], list):
        raise TypeError("PHENOMAP_SPECS must be a list")

    for triple in params["PHENOMAP_SPECS"]:
        # genotype index
        if not 0 <= triple[0] < params["MAX_LIFESPAN"]:
            raise ValueError(
                "Genotype index in PHENOMAP_SPECS must be a number in interval [0, MAX_LIFESPAN]"
            )
        # phenotype index
        if not 0 <= triple[1] < params["MAX_LIFESPAN"]:
            raise ValueError(
                "Phenotype index in PHENOMAP_SPECS must be a number in interval [0, MAX_LIFESPAN]"
            )
        # phenomap weight
        if not isinstance(triple[2], (int, float)):
            raise TypeError(
                "Phenomap weight in PHENOMAP_SPECS must be an integer or a float"
            )

    # Environment
    if not isinstance(params["ENVIRONMENT_CHANGE_RATE"], int):
        raise TypeError("ENVIRONMENT_CHANGE_RATE must be an integer")

    if params["ENVIRONMENT_CHANGE_RATE"] < 0:
        raise ValueError("ENVIRONMENT_CHANGE_RATE must be 0 or greater than 0")
