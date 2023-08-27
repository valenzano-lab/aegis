import logging


class Param:
    def __init__(self, name, default, info, dtype, drange):

        self.name = name
        self.default = default
        self.info = info
        self.dtype = dtype
        self.drange = drange

    def convert(self, value):
        if value is None or value == "":
            return self.default
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


params = {
    "RANDOM_SEED_": Param(
        name="RANDOM_SEED_",
        default=None,
        info="If nothing is given, a random integer will be used as the seed; otherwise the given integer will be used as the seed",
        dtype=int,
        drange="int, none",
        inrange=lambda x: True,
    ),
    "STAGES_PER_SIMULATION_": Param(
        name="STAGES_PER_SIMULATION_",
        default=100000,
        info="How many stages does the simulation run for?",
        dtype=int,
        drange="1+",
        inrange=lambda x: x >= 1,
    ),
    "LOGGING_RATE_": Param(
        name="LOGGING_RATE_",
        default=1000,
        info="Log every ?-th stage; 0 for no logging",
        dtype=int,
        drange="0+",
        inrange=lambda x: x >= 0,
    ),
    "PICKLE_RATE_": Param(
        name="PICKLE_RATE_",
        default=50000,
        info="Pickle population every ? stages; 0 for no pickles",
        dtype=int,
        drange="0+",
        inrange=lambda x: x >= 0,
    ),
    "SNAPSHOT_RATE_": Param(
        name="SNAPSHOT_RATE_",
        default=25000,
        info="Take a snapshot every ? stages; 0 for no snapshots",
        dtype=int,
        drange="0+",
        inrange=lambda x: x >= 0,
    ),
    "VISOR_RATE_": Param(
        name="VISOR_RATE_",
        default=5000,
        info="Take a visor snapshot every ? stages; 0 for no visor records",
        dtype=int,
        drange="0+",
        inrange=lambda x: x >= 0,
    ),
    "POPGENSTATS_RATE_": Param(
        name="POPGENSTATS_RATE_",
        default=5000,
        info="Record population genetic stats about the population every ? stages; 0 for no popgen stats",
        dtype=int,
        drange="0+",
        inrange=lambda x: x >= 0,
    ),
    "POPGENSTATS_SAMPLE_SIZE_": Param(
        name="POPGENSTATS_SAMPLE_SIZE_",
        default=100,
        info="Number of individuals to use when calculating population genetic metrics",
        dtype=int,
        drange="0, 3+",
        inrange=lambda x: x == 0 or x >= 3,
    ),
    "ECOSYSTEM_NUMBER_": Param(
        name="ECOSYSTEM_NUMBER_",
        default=1,
        info="",
        dtype=int,
        drange="",
        inrange=lambda x: x >= 1,
    ),
    "MAX_POPULATION_SIZE": Param(
        name="MAX_POPULATION_SIZE",
        default=300,
        info="Number of individuals in the population",
        dtype=int,
        drange="1+",
        inrange=lambda x: x >= 1,
    ),
    "OVERSHOOT_EVENT": Param(
        name="OVERSHOOT_EVENT",
        default="starvation",
        info="Who dies when everyone is starving?",
        dtype=str,
        drange="starvation, cliff, treadmill_random, treadmill_zoomer, treadmill_boomer",
        inrange=lambda x: x
        in (
            "starvation",
            "cliff",
            "treadmill_random",
            "treadmill_zoomer",
            "treadmill_boomer",
        ),
    ),
    "CLIFF_SURVIVORSHIP": Param(
        name="CLIFF_SURVIVORSHIP",
        default=None,
        info="What fraction of population survives after a cliff?; null if not applicable",
        dtype=int,
        drange="null or 0-1 excluding 0 and 1",
        inrange=lambda x: x is None or (0 < x < 1),
    ),
    "STAGES_PER_SEASON": Param(
        name="STAGES_PER_SEASON",
        default=0,
        info="How many stages does one season last; 0 for no seasons",
        dtype=int,
        drange="0+",
        inrange=lambda x: x >= 0,
    ),
    "MAX_LIFESPAN": Param(
        name="MAX_LIFESPAN",
        default=50,
        info="Maximum lifespan",
        dtype=int,
        drange="1+",
        inrange=lambda x: x >= 1,
    ),
    "MATURATION_AGE": Param(
        name="MATURATION_AGE",
        default=10,
        info="Age at which reproduction is possible",
        dtype=int,
        drange="1+",
        inrange=lambda x: x >= 1,
    ),
    "BITS_PER_LOCUS": Param(
        name="BITS_PER_LOCUS",
        default=8,
        info="Number of bits that each locus has",
        dtype=int,
        drange="1+",
        inrange=lambda x: x >= 1,
    ),
    "HEADSUP": Param(
        name="HEADSUP",
        default=-1,
        info="-1 if no preevolution, 0 for maturity guarantee, +x for headsup",
        dtype=int,
        drange="-1, 0, 1+",
        inrange=lambda x: x in (-1, 0) or x >= 1,
    ),
    "REPRODUCTION_MODE": Param(
        name="REPRODUCTION_MODE",
        default="asexual",
        info="Mode of reproduction",
        dtype=str,
        drange="sexual, asexual, asexual_diploid",
        inrange=lambda x: x in ("sexual", "asexual", "asexual_diploid"),
    ),
    "RECOMBINATION_RATE": Param(
        name="RECOMBINATION_RATE",
        default=0,
        info="Rate of recombination; 0 if no recombination",
        dtype=float,
        drange="0+",
        inrange=lambda x: x >= 0,
    ),
    "MUTATION_RATIO": Param(
        name="MUTATION_RATIO",
        default=0.1,
        info="Ratio of 0->1 mutations to 1->0 mutations",
        dtype=float,
        drange="0+",
        inrange=lambda x: x >= 0,
    ),
    "MUTATION_METHOD": Param(
        name="MUTATION_METHOD",
        default="by_bit",
        info="",
        dtype=str,
        drange="by_bit, by_index",
        inrange=lambda x: x in ("by_bit", "by_index"),
    ),
    "PHENOMAP_METHOD": Param(
        name="PHENOMAP_METHOD",
        default="by_loop",
        info="Non-vectorized, vectorized and null method of calculating phenotypic values",
        dtype=str,
        drange="by_loop, by_dot, by_dummy",
        inrange=lambda x: x in ("by_loop", "by_dot", "by_dummy"),
    ),
    "ENVIRONMENT_CHANGE_RATE": Param(
        name="ENVIRONMENT_CHANGE_RATE",
        default=0,
        info="Environmental map changes every ? stages; if no environmental change",
        dtype=int,
        drange="0+",
        inrange=lambda x: x >= 0,
    ),
    "G_surv_evolvable": Param(
        name="G_surv_evolvable",
        default=True,
        info="",
        dtype=bool,
        drange="",
        inrange=lambda x: True,
    ),
    "G_surv_agespecific": Param(
        name="G_surv_agespecific",
        default=True,
        info="",
        dtype=bool,
        drange="",
        inrange=lambda x: True,
    ),
    "G_surv_interpreter": Param(
        name="G_surv_interpreter",
        default="binary",
        info="",
        dtype=str,
        drange="",
    ),
    "G_surv_lo": Param(
        name="G_surv_lo",
        default=0,
        info="",
        dtype=float,
        drange="",
    ),
    "G_surv_hi": Param(
        name="G_surv_hi",
        default=1,
        info="",
        dtype=float,
        drange="",
    ),
    "G_surv_initial": Param(
        name="G_surv_initial",
        default=1,
        info="",
        dtype=float,
        drange="",
    ),
    "G_repr_evolvable": Param(
        name="G_repr_evolvable",
        default=True,
        info="",
        dtype=bool,
        drange="",
    ),
    "G_repr_agespecific": Param(
        name="G_repr_agespecific",
        default=True,
        info="",
        dtype=bool,
        drange="",
    ),
    "G_repr_interpreter": Param(
        name="G_repr_interpreter",
        default="binary",
        info="",
        dtype=str,
        drange="",
    ),
    "G_repr_lo": Param(
        name="G_repr_lo",
        default=0,
        info="",
        dtype=float,
        drange="",
    ),
    "G_repr_hi": Param(
        name="G_repr_hi",
        default=0.5,
        info="",
        dtype=float,
        drange="",
    ),
    "G_repr_initial": Param(
        name="G_repr_initial",
        default=1,
        info="",
        dtype=float,
        drange="",
    ),
    "G_neut_evolvable": Param(
        name="G_neut_evolvable",
        default=False,
        info="",
        dtype=bool,
        drange="",
    ),
    "G_neut_agespecific": Param(
        name="G_neut_agespecific",
        default=None,
        info="",
        dtype=bool,
        drange="",
    ),
    "G_neut_interpreter": Param(
        name="G_neut_interpreter",
        default=None,
        info="",
        dtype=str,
        drange="",
    ),
    "G_neut_lo": Param(
        name="G_neut_lo",
        default=None,
        info="",
        dtype=float,
        drange="",
    ),
    "G_neut_hi": Param(
        name="G_neut_hi",
        default=None,
        info="",
        dtype=float,
        drange="",
    ),
    "G_neut_initial": Param(
        name="G_neut_initial",
        default=1,
        info="",
        dtype=float,
        drange="",
    ),
    "G_muta_evolvable": Param(
        name="G_muta_evolvable",
        default=False,
        info="",
        dtype=bool,
        drange="",
    ),
    "G_muta_agespecific": Param(
        name="G_muta_agespecific",
        default=None,
        info="",
        dtype=bool,
        drange="",
    ),
    "G_muta_interpreter": Param(
        name="G_muta_interpreter",
        default=None,
        info="",
        dtype=str,
        drange="",
    ),
    "G_muta_lo": Param(
        name="G_muta_lo",
        default=None,
        info="",
        dtype=float,
        drange="",
    ),
    "G_muta_hi": Param(
        name="G_muta_hi",
        default=None,
        info="",
        dtype=float,
        drange="",
    ),
    "G_muta_initial": Param(
        name="G_muta_initial",
        default=0.001,
        info="",
        dtype=float,
        drange="",
    ),
    "PHENOMAP_SPECS": Param(
        name="PHENOMAP_SPECS",
        default=[],
        info="",
        dtype=list,
        drange="",
    ),
    "NOTES": Param(
        name="NOTES",
        default=[],
        info="",
        dtype=list,
        drange="",
    ),
}
