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


