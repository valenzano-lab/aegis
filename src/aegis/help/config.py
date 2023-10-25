import logging


def get_default_parameters():
    return {p.key: p.default for p in params.values()}


def validate(pdict):
    for key, val in pdict.items():
        # Validate key
        if all(key != p.key for p in params.values()):
            raise ValueError(f"'{key}' is not a valid parameter name")

        # Validate value type and range
        params[key].validate_dtype(val)
        params[key].validate_inrange(val)


class Param:
    def __init__(
        self, key, name, domain, default, info, dtype, drange, inrange=lambda x: True
    ):
        self.key = key
        self.name = name
        self.domain = domain
        self.default = default
        self.info = info
        self.dtype = dtype
        self.drange = drange
        self.inrange = inrange

    def convert(self, value):
        if value is None or value == "":
            return self.default
        elif self.dtype == bool:
            return value == "True" or value == "true"
        else:
            return self.dtype(value)

    def valid(self, value):
        # Not valid if wrong data type
        if not isinstance(value, self.dtype):
            logging.error(
                f"Value {value} is not of valid type {self.dtype} but of type {type(value)}"
            )
            return False

        # Not valid if not in range
        if not self.inrange(value):
            return False

        # Valid
        return True

    def get_name(self):
        if self.name:
            return self.name
        name = self.key.replace("_", " ").strip().lower()
        return name

    def validate_dtype(self, value):
        can_be_none = self.default is None
        # given custom value is None which is a valid data type
        if can_be_none and value is None:
            return
        # given custom value is of valid data type
        if isinstance(value, self.dtype):
            return
        # given custom value is int but float is valid
        if self.dtype is float and isinstance(value, int):
            return
        raise TypeError(
            f"You set {self.key} to be '{value}' which is of type {type(value)} but it should be {self.dtype} {'or None' if can_be_none else ''}"
        )

    def validate_inrange(self, value):
        if self.inrange(value):
            return
        raise ValueError(
            f"{self.key} is set to be '{value}' which is outside of the valid range '{self.drange}'."
        )


# You need the keys so you can find the param (in a list, you cannot)
params = {
    # "": Param(
    #     key="",
    #     name="",
    #     domain="",
    #     default=None,
    #     info="",
    #     dtype=float,
    #     drange="[0,inf)",
    #     inrange=lambda x: x >= 0,
    # ),
    "RANDOM_SEED_": Param(
        key="RANDOM_SEED_",
        name="",
        domain="recording",
        default=None,
        info="If nothing is given, a random integer will be used as the seed; otherwise the given integer will be used as the seed",
        dtype=int,
        drange="{None, (-inf, inf)}",
        inrange=lambda x: True,
    ),
    "STAGES_PER_SIMULATION_": Param(
        key="STAGES_PER_SIMULATION_",
        name="",
        domain="recording",
        default=100000,
        info="How many stages does the simulation run for?",
        dtype=int,
        drange="[1, inf)",
        inrange=lambda x: x >= 1,
    ),
    "LOGGING_RATE_": Param(
        key="LOGGING_RATE_",
        name="",
        domain="recording",
        default=1000,
        info="Log every ?-th stage; 0 for no logging",
        dtype=int,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "PICKLE_RATE_": Param(
        key="PICKLE_RATE_",
        name="",
        domain="recording",
        default=100000,
        info="Pickle population every ? stages; 0 for no pickles",
        dtype=int,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "SNAPSHOT_RATE_": Param(
        key="SNAPSHOT_RATE_",
        name="",
        domain="recording",
        default=10000,
        info="Take a snapshot every ? stages; 0 for no snapshots",
        dtype=int,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "VISOR_RATE_": Param(
        key="VISOR_RATE_",
        name="",
        domain="recording",
        default=1000,
        info="Take a visor snapshot every ? stages; 0 for no visor records",
        dtype=int,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "POPGENSTATS_RATE_": Param(
        key="POPGENSTATS_RATE_",
        name="",
        domain="recording",
        default=1000,
        info="Record population genetic stats about the population every ? stages; 0 for no popgen stats",
        dtype=int,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "POPGENSTATS_SAMPLE_SIZE_": Param(
        key="POPGENSTATS_SAMPLE_SIZE_",
        name="",
        domain="recording",
        default=100,
        info="Number of individuals to use when calculating population genetic metrics",
        dtype=int,
        drange="{0, [3, inf)}",
        inrange=lambda x: x == 0 or x >= 3,
    ),
    "ECOSYSTEM_NUMBER_": Param(
        key="ECOSYSTEM_NUMBER_",
        name="",
        domain="ecology",
        default=1,
        info="Number of subpopulations",
        dtype=int,
        drange="[1, inf)",
        inrange=lambda x: x >= 1,
    ),
    "MAX_POPULATION_SIZE": Param(
        key="MAX_POPULATION_SIZE",
        name="",
        domain="ecology",
        default=1000,
        info="Number of individuals in the population",
        dtype=int,
        drange="[1, inf)",
        inrange=lambda x: x >= 1,
    ),
    "OVERSHOOT_EVENT": Param(
        key="OVERSHOOT_EVENT",
        name="",
        domain="ecology",
        default="starvation",
        info="Who dies when everyone is starving?",
        dtype=str,
        drange="{starvation, cliff, treadmill_random, treadmill_zoomer, treadmill_boomer}",
        inrange=lambda x: x
        in (
            "starvation",
            "cliff",
            "treadmill_random",
            "treadmill_zoomer",
            "treadmill_boomer",
        ),
    ),
    "OVERSHOOT_MORTALITY": Param(
        key="OVERSHOOT_MORTALITY",
        name="",
        domain="ecology",
        default=0.05,
        info="",
        dtype=float,
        drange="(0,1)",
        inrange=lambda x: 0 <= x <= 1,
    ),
    "CLIFF_SURVIVORSHIP": Param(
        key="CLIFF_SURVIVORSHIP",
        name="",
        domain="ecology",
        default=None,
        info="What fraction of population survives after a cliff?; null if not applicable",
        dtype=float,
        drange="{None, (0,1)}",
        inrange=lambda x: x is None or (0 < x < 1),
    ),
    "STAGES_PER_SEASON": Param(
        key="STAGES_PER_SEASON",
        name="",
        domain="ecology",
        default=0,
        info="How many stages does one season last; 0 for no seasons",
        dtype=int,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "MAX_LIFESPAN": Param(
        key="MAX_LIFESPAN",
        name="",
        domain="genetics",
        default=50,
        info="Maximum lifespan",
        dtype=int,
        drange="[1, inf)",
        inrange=lambda x: x >= 1,
    ),
    "MATURATION_AGE": Param(
        key="MATURATION_AGE",
        name="",
        domain="genetics",
        default=10,
        info="Age at which reproduction is possible",
        dtype=int,
        drange="[1, inf)",
        inrange=lambda x: x >= 1,
    ),
    "BITS_PER_LOCUS": Param(
        key="BITS_PER_LOCUS",
        name="",
        domain="genetics",
        default=8,
        info="Number of bits that each locus has",
        dtype=int,
        drange="[1, inf)",
        inrange=lambda x: x >= 1,
    ),
    "HEADSUP": Param(
        key="HEADSUP",
        name="",
        domain="initialization",
        default=-1,
        info="-1 if no preevolution, 0 for maturity guarantee, +x for headsup",
        dtype=int,
        drange="{-1, 0, [1, inf)}",
        inrange=lambda x: x in (-1, 0) or x >= 1,
    ),
    "REPRODUCTION_MODE": Param(
        key="REPRODUCTION_MODE",
        name="",
        domain="genetics",
        default="asexual",
        info="Mode of reproduction",
        dtype=str,
        drange="{sexual, asexual, asexual_diploid}",
        inrange=lambda x: x in ("sexual", "asexual", "asexual_diploid"),
    ),
    "RECOMBINATION_RATE": Param(
        key="RECOMBINATION_RATE",
        name="",
        domain="genetics",
        default=0,
        info="Rate of recombination; 0 if no recombination",
        dtype=float,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "MUTATION_RATIO": Param(
        key="MUTATION_RATIO",
        name="",
        domain="genetics",
        default=0.1,
        info="Ratio of 0->1 mutations to 1->0 mutations",
        dtype=float,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "MUTATION_METHOD": Param(
        key="MUTATION_METHOD",
        name="",
        domain="computation",
        default="by_bit",
        info="Mutate by XOR with a randomized bit matrix or generate random indices to mutate",
        dtype=str,
        drange="{by_bit, by_index}",
        inrange=lambda x: x in ("by_bit", "by_index"),
    ),
    "DOMINANCE_FACTOR": Param(
        key="DOMINANCE_FACTOR",
        name="",
        domain="",
        default=0.5,
        info="0 for recessive, 0.5 for true additive, 0-1 for partial dominant, 1 for dominant, 1+ for overdominant",
        dtype=float,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "PHENOMAP_METHOD": Param(
        key="PHENOMAP_METHOD",
        name="",
        domain="computation",
        default="by_loop",
        info="Non-vectorized, vectorized and null method of calculating phenotypic values",
        dtype=str,
        drange="{by_loop, by_dot, by_dummy}",
        inrange=lambda x: x in ("by_loop", "by_dot", "by_dummy"),
    ),
    "FLIPMAP_CHANGE_RATE": Param(
        key="FLIPMAP_CHANGE_RATE",
        name="",
        domain="ecology",
        default=0,
        info="Flipmap changes every ? stages",
        dtype=int,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "ENVIRONMENT_HAZARD_AMPLITUDE": Param(
        key="ENVIRONMENT_HAZARD_AMPLITUDE",
        name="",
        domain="environment",
        default=0,
        info="",
        dtype=float,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "ENVIRONMENT_HAZARD_PERIOD": Param(
        key="ENVIRONMENT_HAZARD_PERIOD",
        name="",
        domain="environment",
        default=1,
        info="",
        dtype=float,
        drange="[1, inf)",
        inrange=lambda x: x >= 1,
    ),
    "ENVIRONMENT_HAZARD_OFFSET": Param(
        key="ENVIRONMENT_HAZARD_OFFSET",
        name="",
        domain="environment",
        default=0,
        info=r"e.g. 0.01 means that environmental mortality is increased by 1% each stage",
        dtype=float,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "ENVIRONMENT_HAZARD_SHAPE": Param(
        key="ENVIRONMENT_HAZARD_SHAPE",
        name="",
        domain="environment",
        default="sinusoidal",
        info="",
        dtype=str,
        drange="{sinusoidal, flat, triangle, square, sawtooth, ramp}",
        inrange=lambda x: x
        in {"sinusoidal", "flat", "triangle", "square", "sawtooth", "ramp"},
    ),
    "BACKGROUND_INFECTIVITY": Param(
        key="BACKGROUND_INFECTIVITY",
        name="",
        domain="infection",
        default=0,
        info="",
        dtype=float,
        drange="[0,inf)",
        inrange=lambda x: x >= 0,
    ),
    "TRANSMISSIBILITY": Param(
        key="TRANSMISSIBILITY",
        name="",
        domain="",
        default=0,
        info="",
        dtype=float,
        drange="[0,inf)",
        inrange=lambda x: x >= 0,
    ),
    "RECOVERY_RATE": Param(
        key="RECOVERY_RATE",
        name="",
        domain="",
        default=0,
        info="",
        dtype=float,
        drange="[0,inf)",
        inrange=lambda x: x >= 0,
    ),
    "FATALITY_RATE": Param(
        key="FATALITY_RATE",
        name="",
        domain="",
        default=0,
        info="",
        dtype=float,
        drange="[0,inf)",
        inrange=lambda x: x >= 0,
    ),
    "PREDATION_RATE": Param(
        key="PREDATION_RATE",
        name="",
        domain="",
        default=0,
        info="Mortality rate when number of predators equal to number of prey",
        dtype=float,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "PREDATOR_GROWTH": Param(
        key="PREDATOR_GROWTH",
        name="",
        domain="",
        default=0,
        info="Intrinsic growth rate of predators",
        dtype=float,
        drange="[0,inf)",
        inrange=lambda x: x >= 0,
    ),
    "G_surv_evolvable": Param(
        key="G_surv_evolvable",
        name="",
        domain="genetics",
        default=True,
        info="Is survival an evolvable trait?",
        dtype=bool,
        drange="",
        inrange=lambda x: True,
    ),
    "G_surv_agespecific": Param(
        key="G_surv_agespecific",
        name="",
        domain="genetics",
        default=True,
        info="Is survival age-specific?",
        dtype=bool,
        drange="",
        inrange=lambda x: True,
    ),
    "G_surv_interpreter": Param(
        key="G_surv_interpreter",
        name="",
        domain="genetics",
        default="binary",
        info="",
        dtype=str,
        drange="",
    ),
    "G_surv_lo": Param(
        key="G_surv_lo",
        name="",
        domain="genetics",
        default=0,
        info="Minimum survival rate",
        dtype=float,
        drange="",
    ),
    "G_surv_hi": Param(
        key="G_surv_hi",
        name="",
        domain="genetics",
        default=1,
        info="Maximum survival rate",
        dtype=float,
        drange="",
    ),
    "G_surv_initial": Param(
        key="G_surv_initial",
        name="",
        domain="initialization",
        default=1,
        info="Initial survival rate",
        dtype=float,
        drange="",
    ),
    "G_repr_evolvable": Param(
        key="G_repr_evolvable",
        name="",
        domain="genetics",
        default=True,
        info="Is fertility an evolvable trait?",
        dtype=bool,
        drange="",
    ),
    "G_repr_agespecific": Param(
        key="G_repr_agespecific",
        name="",
        domain="genetics",
        default=True,
        info="Is fertility age-specific?",
        dtype=bool,
        drange="",
    ),
    "G_repr_interpreter": Param(
        key="G_repr_interpreter",
        name="",
        domain="genetics",
        default="binary",
        info="",
        dtype=str,
        drange="",
    ),
    "G_repr_lo": Param(
        key="G_repr_lo",
        name="",
        domain="genetics",
        default=0,
        info="Minimum fertility",
        dtype=float,
        drange="",
    ),
    "G_repr_hi": Param(
        key="G_repr_hi",
        name="",
        domain="genetics",
        default=0.5,
        info="Maximum fertility",
        dtype=float,
        drange="",
    ),
    "G_repr_initial": Param(
        key="G_repr_initial",
        name="",
        domain="initialization",
        default=1,
        info="Initial fertility rate",
        dtype=float,
        drange="",
    ),
    "G_neut_evolvable": Param(
        key="G_neut_evolvable",
        name="",
        domain="genetics",
        default=False,
        info="",
        dtype=bool,
        drange="",
    ),
    "G_neut_agespecific": Param(
        key="G_neut_agespecific",
        name="",
        domain="genetics",
        default=None,
        info="",
        dtype=bool,
        drange="",
    ),
    "G_neut_interpreter": Param(
        key="G_neut_interpreter",
        name="",
        domain="genetics",
        default=None,
        info="",
        dtype=str,
        drange="",
    ),
    "G_neut_lo": Param(
        key="G_neut_lo",
        name="",
        domain="genetics",
        default=None,
        info="",
        dtype=float,
        drange="",
    ),
    "G_neut_hi": Param(
        key="G_neut_hi",
        name="",
        domain="genetics",
        default=None,
        info="",
        dtype=float,
        drange="",
    ),
    "G_neut_initial": Param(
        key="G_neut_initial",
        name="",
        domain="initialization",
        default=1,
        info="",
        dtype=float,
        drange="",
    ),
    "G_muta_evolvable": Param(
        key="G_muta_evolvable",
        name="",
        domain="genetics",
        default=False,
        info="",
        dtype=bool,
        drange="",
    ),
    "G_muta_agespecific": Param(
        key="G_muta_agespecific",
        name="",
        domain="genetics",
        default=None,
        info="",
        dtype=bool,
        drange="",
    ),
    "G_muta_interpreter": Param(
        key="G_muta_interpreter",
        name="",
        domain="genetics",
        default=None,
        info="",
        dtype=str,
        drange="",
    ),
    "G_muta_lo": Param(
        key="G_muta_lo",
        name="",
        domain="genetics",
        default=None,
        info="",
        dtype=float,
        drange="",
    ),
    "G_muta_hi": Param(
        key="G_muta_hi",
        name="",
        domain="genetics",
        default=None,
        info="",
        dtype=float,
        drange="",
    ),
    "G_muta_initial": Param(
        key="G_muta_initial",
        name="",
        domain="initialization",
        default=0.001,
        info="Initial mutation rate",
        dtype=float,
        drange="",
    ),
    "THRESHOLD": Param(
        key="THRESHOLD",
        name="",
        domain="genetics",
        default=None,  # 3
        info="",
        dtype=int,
        drange="",
    ),
    "PHENOMAP_SPECS": Param(
        key="PHENOMAP_SPECS",
        name="",
        domain="genetics",
        default=[],
        info="",
        dtype=list,
        drange="",
    ),
    "NOTES": Param(
        key="NOTES",
        name="",
        domain="recording",
        default=[],
        info="",
        dtype=list,
        drange="",
    ),
}
