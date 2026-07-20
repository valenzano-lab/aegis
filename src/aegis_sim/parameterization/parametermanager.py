import logging
import yaml
import types

from aegis_sim.parameterization.default_parameters import (
    get_default_parameters,
    DEFAULT_PARAMETERS,
    get_species_parameters,
)


class ParameterManager:
    def init(self, custom_config_path, custom_input_params):
        self.custom_config_path = custom_config_path
        self.custom_input_params = custom_input_params
        self.final_config = None

        self.parameters = self()

    #: Parameters that define the SHAPE of the checkpointed genome, phenotype and
    #: population arrays. Overriding any of these on resume would leave the restored
    #: arrays structurally inconsistent with the parameters describing them -- e.g.
    #: raising AGE_LIMIT would make trait slices index past the end of a genome that
    #: was built for the old value. Rejected loudly rather than silently corrupting.
    STRUCTURAL_PARAMETERS = frozenset(
        {
            "AGE_LIMIT",
            "BITS_PER_LOCUS",
            "PLOIDY",
            "GENARCH_TYPE",
            "REPRODUCTION_MODE",
            "MODIF_GENOME_SIZE",
        }
        | {f"G_{trait}_{attr}" for trait in ("surv", "repr", "muta", "neut", "grow") for attr in ("evolvable", "agespecific")}
        #: not structural, but the RNG state is restored from the checkpoint, so a new
        #: seed would be silently ignored -- reject rather than mislead.
        | {"RANDOM_SEED"}
    )

    def init_from_config(self, final_config, custom_config_path, overrides=None):
        """Initialize from a saved config dict (used when resuming from checkpoint).

        `overrides` applies parameter changes on top of the checkpointed config. This
        is what makes a two-phase experiment expressible: burn a population in to
        equilibrium under one regime, then resume the SAME checkpoint several times
        under different regimes, so every arm shares an identical equilibrated
        ancestor and between-arm differences cannot come from the burn-in.

        Only non-structural parameters may be overridden; see STRUCTURAL_PARAMETERS.
        """
        self.custom_config_path = custom_config_path
        self.custom_input_params = {}
        self.final_config = final_config.copy()

        if overrides:
            self.validate(overrides)
            illegal = sorted(set(overrides) & self.STRUCTURAL_PARAMETERS)
            if illegal:
                raise ValueError(
                    f"Cannot override {illegal} on resume: these define the shape of the "
                    f"checkpointed genome/phenotype arrays (or, for RANDOM_SEED, are "
                    f"superseded by the restored RNG state). Start a fresh run instead."
                )
            for key, value in overrides.items():
                old = self.final_config.get(key)
                self.final_config[key] = value
                logging.info(f"Resume override: {key} {old!r} -> {value!r}")

        self.parameters = types.SimpleNamespace(**self.final_config)
        logging.info("Parameters restored from checkpoint.")

    def __call__(self):
        """
        Getting parameters from three sources:
        1. Default
        2. Configuration file
        3. Function arguments

        When a parameter value is specified multiple times, 3 overwrites 2 which overwrites 1.
        """

        default_parameters = get_default_parameters()
        custom_config_params = self.read_config_file()
        self.validate(custom_config_params)
        for k in default_parameters.keys():
            if k in custom_config_params and default_parameters[k] != custom_config_params[k]:
                logging.debug(
                    f"-- {k} is different in config ({custom_config_params[k]}) vs default ({default_parameters[k]})"
                )

        SPECIES_PRESET = custom_config_params.get("SPECIES_PRESET", default_parameters["SPECIES_PRESET"])
        species_config_params = get_species_parameters(SPECIES_PRESET)

        logging.info(f"Using {SPECIES_PRESET} as species preset: " + repr(species_config_params) + ".")

        # Fuse
        params = {}
        params.update(default_parameters)
        params.update(species_config_params)
        params.update(custom_config_params)
        params.update(self.custom_input_params)

        self.final_config = params.copy()

        # convert to types.SimpleNamespace
        params = types.SimpleNamespace(**params)
        logging.info("Final parameters to use in the simulation: " + repr(params) + ".")
        return params

    def read_config_file(self):

        # No configuration file specified
        if self.custom_config_path == "":
            logging.info("No configuration file has been specified.")
            return {}

        # Configuration file specified...
        with open(self.custom_config_path, "r") as file_:
            ccp = yaml.safe_load(file_)

        # ... but it is empty
        if ccp is None:
            logging.info("Configuration file is empty.")
            ccp = {}

        return ccp

    @staticmethod
    def validate(pdict, validate_serverrange=False):
        for key, val in pdict.items():
            # Validate key
            if all(key != p.key for p in DEFAULT_PARAMETERS.values()):
                raise ValueError(f"'{key}' is not a valid parameter name")

            # Validate value type and range
            DEFAULT_PARAMETERS[key].validate_dtype(val)
            DEFAULT_PARAMETERS[key].validate_inrange(val)

            if validate_serverrange:
                DEFAULT_PARAMETERS[key].validate_serverrange(val)
