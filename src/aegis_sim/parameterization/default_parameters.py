from aegis_sim.parameterization.parameter import Parameter


def get_default_parameters():
    return {p.key: p.default for p in DEFAULT_PARAMETERS.values()}


def get_species_parameters(SPECIES_PRESET):
    return {p.key: p.presets[SPECIES_PRESET] for p in DEFAULT_PARAMETERS.values() if SPECIES_PRESET in p.presets}


# TODO test these
# TODO value interpolation between ages?
# TODO report invalid values

PRESET_INFO = {
    "human": "One cycle corresponds to 2 years.",
    "mouse": "One cycle corresponds to one month. Source: https://genomics.senescence.info/species/entry.php?species=Mus_musculus",
    "killifish": "One cycle corresponds to one week.",
    "yeast": "",
    "arabidopsis": "",
    "worm": "One cycle corresponds to one day. Up to 300 eggs in optimal conditions.",
    "fruitfly": "One cycle corresponds to one day. Up to 100 eggs per day.",
}

# You need the `key` attribute so you can find the param (in a list, you cannot)
DEFAULT_PARAMETERS = {
    #
    #
    # RECORDING
    "LOGGING_RATE": Parameter(
        key="LOGGING_RATE",
        name="",
        domain="recording",
        default=100,
        info="Frequency of logging (in steps)",
        info_extended="Log files contain information on simulation execution; e.g. errors and speed. 0 for no logging.",
        dtype=int,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
        evalrange=[1, 10000],
    ),
    "TICKER_RATE": Parameter(
        key="TICKER_RATE",
        name="",
        domain="recording",
        default=1,
        info="Frequency of ticking (in seconds)",
        info_extended="Ticker files contain information on simulation status; e.g. running or finished. Once the simulation is finished, it stops updating the ticker file which indicates the time at which the simulation stopped.",
        dtype=float,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
        evalrange=[1, 10000],
    ),
    "PICKLE_RATE": Parameter(
        key="PICKLE_RATE",
        name="",
        domain="recording",
        default=10000,
        info="Frequency of pickling (in steps)",
        info_extended="0 for no pickles",
        dtype=int,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
        serverrange=lambda x: x >= 1000 or x == 0,
        serverrange_info="0 or [1000, inf)",
        evalrange=[1, 100000],
    ),
    "SNAPSHOT_RATE": Parameter(
        key="SNAPSHOT_RATE",
        name="",
        domain="recording",
        default=10000,
        info="Frequency of recording snapshots (in steps)",
        info_extended="0 for no snapshots",
        dtype=int,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
        serverrange=lambda x: x >= 1000 or x == 0,
        serverrange_info="0 or [1000, inf)",
        evalrange=[1, 10000],
    ),
    "SNAPSHOT_FINAL_COUNT": Parameter(
        key="SNAPSHOT_FINAL_COUNT",
        name="",
        domain="recording",
        default=60,
        info="Number of subsequent snapshots taken at the end of the simulation (in steps)",
        info_extended="0 for no snapshots",
        dtype=int,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
        serverrange=lambda x: 0 <= x <= 60,
        serverrange_info="[0, 60]",
        evalrange=[1, 10],
    ),
    "INTERVAL_RATE": Parameter(
        key="INTERVAL_RATE",
        name="",
        domain="recording",
        default=1000,
        info="Frequency of recording interval data (in steps)",
        info_extended="0 for no gui records",
        dtype=int,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
        evalrange=[1, 10000],
    ),
    "TE_RATE": Parameter(
        key="TE_RATE",
        name="",
        domain="recording",
        default=10000,
        info="Frequency of starting TE cohorts (in steps)",
        dtype=int,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
        evalrange=[1, 10000],
    ),
    "TE_DURATION": Parameter(
        key="TE_DURATION",
        name="",
        domain="recording",
        default=500,
        info="Length of tracking TE cohorts (in steps)",
        dtype=int,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
        evalrange=[1, 10000],
    ),
    "POPGENSTATS_RATE": Parameter(
        key="POPGENSTATS_RATE",
        name="",
        domain="recording",
        default=1000,
        info="Frequency of recording population genetic statistics (in steps)",
        info_extended="0 for no recording",
        dtype=int,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
        serverrange=lambda x: x >= 100 or x == 0,
        serverrange_info="0 or [100, inf)",
        evalrange=[1, 10000],
    ),
    "POPGENSTATS_SAMPLE_SIZE": Parameter(
        key="POPGENSTATS_SAMPLE_SIZE",
        name="",
        domain="recording",
        default=100,
        info="Number of individuals to use when calculating population genetic statistics",
        dtype=int,
        drange="{0, [3, inf)}",
        inrange=lambda x: x == 0 or x >= 3,
    ),
    "NOTES": Parameter(
        key="NOTES",
        name="",
        domain="recording",
        default=[],
        info="",
        dtype=list,
        drange="",
    ),
    #
    #
    # STARVATION
    "CARRYING_CAPACITY": Parameter(
        key="CARRYING_CAPACITY",
        name="",
        domain="starvation",
        default=500,
        info="Maximum population size that the environment can sustain",
        info_extended="Starvation mortality is incurred when the population exceeds the carrying capacity.",
        dtype=int,
        drange="[1, inf)",
        inrange=lambda x: x >= 1,
        serverrange=lambda x: x <= 10000,
        serverrange_info="[1,100000]",
        evalrange=[1, 1000000],
    ),
    "CARRYING_CAPACITY_EGGS": Parameter(
        key="CARRYING_CAPACITY_EGGS",
        name="",
        domain="starvation",
        default=500,
        info="Maximum number of eggs that the environment can sustain",
        info_extended="Once the number of eggs exceeds the carrying capacity of eggs, newly laid eggs replace previously laid eggs.",
        dtype=int,
        drange="[1, inf)",
        inrange=lambda x: x >= 1,
        serverrange=lambda x: x <= 10000,
        serverrange_info="[1,100000]",
        evalrange=[1, 1000000],
    ),
    "STARVATION_RESPONSE": Parameter(
        key="STARVATION_RESPONSE",
        name="",
        domain="starvation",
        default="gradual",
        info="Mechanism for determining who dies under overcrowding conditions",
        info_extended="The possible modes can differ in the age distribution of mortality and/or the number of individuals removed.",
        dtype=str,
        drange="{gradual, cliff, treadmill_random, treadmill_zoomer, treadmill_boomer, treadmill_boomer_soft, treadmill_zoomer_soft}",
        inrange=lambda x: x
        in (
            "gradual",
            "cliff",
            "treadmill_random",
            "treadmill_zoomer",
            "treadmill_boomer",
            "treadmill_boomer_soft",
            "treadmill_zoomer_soft",
        ),
    ),
    "STARVATION_MAGNITUDE": Parameter(
        key="STARVATION_MAGNITUDE",
        name="",
        domain="starvation",
        default=0.05,
        info="Acceleration of mortality under starvation",
        info_extended="",
        dtype=float,
        drange="(0,1)",
        inrange=lambda x: 0 <= x <= 1,
    ),
    "CLIFF_SURVIVORSHIP": Parameter(
        key="CLIFF_SURVIVORSHIP",
        name="",
        domain="starvation",
        default=None,
        info="Fraction of the population surviving a 'cliff' starvation event",
        info_extended="Modifies the 'cliff' starvation response.",
        dtype=float,
        drange="{None, (0,1)}",
        inrange=lambda x: x is None or (0 < x < 1),
    ),
    #
    #
    # REPRODUCTION
    "INCUBATION_PERIOD": Parameter(
        key="INCUBATION_PERIOD",
        name="",
        domain="reproduction",
        default=0,
        info="Time between fertilization and hatching (in steps)",
        info_extended="0 if egg period is skipped, -1 if hatching occurs only once no living individuals are around.",
        dtype=int,
        drange="[-1, inf)",
        inrange=lambda x: x >= -1,
        presets={},
    ),
    "MATURATION_AGE": Parameter(
        key="MATURATION_AGE",
        name="",
        domain="reproduction",
        default=10,
        info="Minimum age at which an individual can reproduce",
        info_extended="",
        dtype=int,
        drange="[1, inf)",
        inrange=lambda x: x >= 1,
        evalrange=[0, 50],
        presets={
            "mouse": 1,  # 1 cycle .. 1 month
        },
    ),
    "REPRODUCTION_ENDPOINT": Parameter(
        key="REPRODUCTION_ENDPOINT",
        name="",
        domain="reproduction",
        default=0,
        info="Minimum age at which an individual can no longer reproduce",
        info_extended="When set to 0, there is no loss of fertility.",
        dtype=int,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
        presets={
            "human": 50,
        },
    ),
    "MAX_OFFSPRING_NUMBER": Parameter(
        key="MAX_OFFSPRING_NUMBER",
        name="",
        domain="reproduction",
        default=1,
        info="Maximum number of offspring that an individual can produce each step.",
        info_extended="Also known as clutch size, litter size or brood size, depending on the species.",
        dtype=int,
        drange="[1, inf)",
        inrange=lambda x: x >= 1,
        presets={},
    ),
    # TODO split ploidy from reproduction mode
    "REPRODUCTION_MODE": Parameter(
        key="REPRODUCTION_MODE",
        name="",
        domain="reproduction",
        default="sexual",
        info="Mode of reproduction",
        info_extended="",
        dtype=str,
        drange="{sexual, asexual, asexual_diploid}",
        inrange=lambda x: x in ("sexual", "asexual", "asexual_diploid"),
        presets={
            "yeast": "asexual_diploid",
        },
    ),
    "RECOMBINATION_RATE": Parameter(
        key="RECOMBINATION_RATE",
        name="",
        domain="reproduction",
        default=0.1,
        info="Probability of recombination occuring between two adjacent sites",
        info_extended="If set to 0, there is no recombination.",
        dtype=float,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
        evalrange=[0, 1],
        presets={
            "yeast": 0,
        },
    ),
    "MUTATION_RATIO": Parameter(
        key="MUTATION_RATIO",
        name="",
        domain="reproduction",
        default=0.1,
        info="Ratio of 0->1 mutations to 1->0 mutations",
        dtype=float,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "MUTATION_AGE_MULTIPLIER": Parameter(
        key="MUTATION_AGE_MULTIPLIER",
        name="",
        domain="reproduction",
        default=0,
        info="Modifier of germline mutation rate",
        info_extended="final germline mutation rate = intrinsic mutation rate + (1 * age * MUTATION_AGE_MULTIPLIER)",
        dtype=float,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    #
    #
    # GENETICS
    "DOMINANCE_FACTOR": Parameter(
        key="DOMINANCE_FACTOR",
        name="",
        domain="genetics",
        default=1,
        info="Inheritance patterns for non-haploid genomes",
        info_extended="0 for recessive, 0.5 for true additive, 0-1 for partial dominant, 1 for dominant, 1+ for overdominant",
        dtype=float,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "G_repr_lo": Parameter(
        key="G_repr_lo",
        name="",
        domain="genetics",
        default=0,
        info="Minimum intrinsic fertility",
        dtype=float,
        drange="",
    ),
    "G_repr_hi": Parameter(
        key="G_repr_hi",
        name="",
        domain="genetics",
        default=0.5,
        info="Maximum intrinsic fertility",
        dtype=float,
        drange="",
        evalrange=[0.5, 1],
        # presets={
        #     "mouse": 1,  # 3.5; litter size of 7; 5.4 litters per year; https://genomics.senescence.info/species/entry.php?species=Mus_musculus
        #     "human": 1,  # litter size of 1,
        #     "mouse": 1,  # 5.5; litter size of 5-6
        #     "killifish": 1,  # 50; 1x-1xx eggs, depending on species
        #     "yeast": 1,
        #     "athaliana": 1,  # 1xx seeds per plant
        #     "worm": 1,  # up to 300 eggs in optimal conditions
        #     "fruitfly": 1,  # 100; up to 100 eggs per day in optimal conditions
        # },
    ),
    "G_neut_lo": Parameter(
        key="G_neut_lo",
        name="",
        domain="genetics",
        default=0,
        info="",
        dtype=float,
        drange="",
    ),
    "G_neut_hi": Parameter(
        key="G_neut_hi",
        name="",
        domain="genetics",
        default=1,
        info="",
        dtype=float,
        drange="",
    ),
    "G_muta_lo": Parameter(
        key="G_muta_lo",
        name="",
        domain="genetics",
        default=0,
        info="Minumum intrinsic mutation rate",
        dtype=float,
        drange="",
    ),
    "G_muta_hi": Parameter(
        key="G_muta_hi",
        name="",
        domain="genetics",
        default=1,
        info="Maximum intrinsic mutation rate",
        dtype=float,
        drange="",
    ),
    "G_grow_lo": Parameter(
        key="G_grow_lo",
        name="",
        domain="genetics",
        default=0,
        info="Minimum intrinsic growth rate",
        dtype=float,
        drange="",
    ),
    "G_grow_hi": Parameter(
        key="G_grow_hi",
        name="",
        domain="genetics",
        default=1,
        info="Maximum intrinsic growth rate",
        dtype=float,
        drange="",
    ),
    #
    #
    # ENVIRONMENTAL DRIFT
    "ENVDRIFT_RATE": Parameter(
        key="ENVDRIFT_RATE",
        name="",
        domain="environmental drift",
        default=0,
        info="Frequency of modification to the fitness landscape (in steps)",
        dtype=int,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    #
    #
    # ABIOTIC
    "ABIOTIC_HAZARD_AMPLITUDE": Parameter(
        key="ABIOTIC_HAZARD_AMPLITUDE",
        name="",
        domain="abiotic",
        default=0,
        info="Maximum abiotic hazard",
        dtype=float,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "ABIOTIC_HAZARD_PERIOD": Parameter(
        key="ABIOTIC_HAZARD_PERIOD",
        name="",
        domain="abiotic",
        default=1,
        info="Period of wave form of abiotic hazard (in steps)",
        dtype=float,
        drange="[1, inf)",
        inrange=lambda x: x >= 1,
    ),
    "ABIOTIC_HAZARD_OFFSET": Parameter(
        key="ABIOTIC_HAZARD_OFFSET",
        name="",
        domain="abiotic",
        default=0,
        info="Constant, time-independent abiotic hazard",
        info_extended=r"e.g. 0.01 means that abiotic mortality is increased by 1% each step",
        dtype=float,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "ABIOTIC_HAZARD_SHAPE": Parameter(
        key="ABIOTIC_HAZARD_SHAPE",
        name="",
        domain="abiotic",
        default="sinusoidal",
        info="Wave form of abiotic hazard",
        dtype=str,
        drange="{sinusoidal, flat, triangle, square, sawtooth, ramp, instant}",
        inrange=lambda x: x in {"sinusoidal", "flat", "triangle", "square", "sawtooth", "ramp", "instant"},
    ),
    #
    #
    # INFECTION
    "BACKGROUND_INFECTIVITY": Parameter(
        key="BACKGROUND_INFECTIVITY",
        name="",
        domain="infection",
        default=0,
        info="Tendency to acquire infection from the environment",
        info_extended="Probability independent of the infection prevalence in the population; thus constant.",
        dtype=float,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "TRANSMISSIBILITY": Parameter(
        key="TRANSMISSIBILITY",
        name="",
        domain="infection",
        default=0,
        info="Tendency to acquire infection from other infected individuals",
        info_extended="Probability dependent on the infection prevalence in the population; thus variable.",
        dtype=float,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "RECOVERY_RATE": Parameter(
        key="RECOVERY_RATE",
        name="",
        domain="infection",
        info="Tendency to transition from infected to healthy status",
        default=0,
        dtype=float,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "FATALITY_RATE": Parameter(
        key="FATALITY_RATE",
        name="",
        domain="infection",
        info="Tendency to transition from infected to dead status",
        default=0,
        dtype=float,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    #
    #
    # PREDATION
    "PREDATION_RATE": Parameter(
        key="PREDATION_RATE",
        name="",
        domain="predation",
        default=0,
        info="Vulnerability to predators",
        info_extended="Probability to die when number of predators is equal to number of prey. Probability changes logistically with the number of prey.",
        dtype=float,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    "PREDATOR_GROWTH": Parameter(
        key="PREDATOR_GROWTH",
        name="",
        domain="predation",
        default=0,
        info="Intrinsic growth rate of predators",
        info_extended="Growth of the predator population is logistic.",
        dtype=float,
        drange="[0, inf)",
        inrange=lambda x: x >= 0,
    ),
    #
    #
    # GENETIC ARCHITECTURE (COMPOSITE)
    "BITS_PER_LOCUS": Parameter(
        key="BITS_PER_LOCUS",
        name="",
        domain="composite genetic architecture",
        default=8,
        info="Number of bits that each locus has",
        dtype=int,
        drange="[1, inf)",
        inrange=lambda x: x >= 1,
        serverrange=lambda x: x <= 10,
        serverrange_info="[1,10]",
        evalrange=[1, 100],
    ),
    "HEADSUP": Parameter(
        key="HEADSUP",
        name="",
        domain="composite genetic architecture",
        default=-1,
        info="-1 if no preevolution, 0 for maturity guarantee, +x for headsup",
        dtype=int,
        drange="{-1, 0, [1, inf)}",
        inrange=lambda x: x in (-1, 0) or x >= 1,
    ),
    # "DIFFUSION_FACTOR": Parameter(
    #     key="DIFFUSION_FACTOR",
    #     name="",
    #     domain="composite genetic architecture",
    #     default=1,
    #     info="Window for moving average",
    #     info_extended="When 1, all variants affect one age and trait only. When 1+, they also affect adjacent ages.",
    #     dtype=int,
    #     drange="[1, inf)",
    #     inrange=lambda x: x >= 1,
    #     serverrange=lambda x: x <= 10,
    #     serverrange_info="[1,10]",
    #
    # evalrange=[1, 50],
    # ),
    "G_surv_evolvable": Parameter(
        key="G_surv_evolvable",
        name="",
        domain="composite genetic architecture",
        default=True,
        info="Is survival an evolvable trait?",
        dtype=bool,
        drange="",
        inrange=lambda x: True,
    ),
    "G_surv_agespecific": Parameter(
        key="G_surv_agespecific",
        name="",
        domain="composite genetic architecture",
        default=True,
        info="Is survival age-specific?",
        dtype=bool,
        drange="",
        inrange=lambda x: True,
    ),
    "G_surv_interpreter": Parameter(
        key="G_surv_interpreter",
        name="",
        domain="composite genetic architecture",
        default="binary",
        info="",
        dtype=str,
        drange="",
    ),
    "G_surv_lo": Parameter(
        key="G_surv_lo",
        name="",
        domain="composite genetic architecture",
        default=0,
        info="Minimum survival rate",
        dtype=float,
        drange="",
    ),
    "G_surv_hi": Parameter(
        key="G_surv_hi",
        name="",
        domain="composite genetic architecture",
        default=1,
        info="Maximum survival rate",
        dtype=float,
        drange="",
    ),
    "G_surv_initgeno": Parameter(
        key="G_surv_initgeno",
        name="",
        domain="composite genetic architecture",
        default=1,
        info="Initial survival rate",
        dtype=float,
        drange="",
    ),
    "G_repr_evolvable": Parameter(
        key="G_repr_evolvable",
        name="",
        domain="composite genetic architecture",
        default=True,
        info="Is fertility an evolvable trait?",
        dtype=bool,
        drange="",
    ),
    "G_repr_agespecific": Parameter(
        key="G_repr_agespecific",
        name="",
        domain="composite genetic architecture",
        default=True,
        info="Is fertility age-specific?",
        dtype=bool,
        drange="",
    ),
    "G_repr_interpreter": Parameter(
        key="G_repr_interpreter",
        name="",
        domain="composite genetic architecture",
        default="binary",
        info="",
        dtype=str,
        drange="",
    ),
    "G_repr_initgeno": Parameter(
        key="G_repr_initgeno",
        name="",
        domain="composite genetic architecture",
        default=1,
        info="Initial fertility rate",
        dtype=float,
        drange="",
    ),
    "G_neut_evolvable": Parameter(
        key="G_neut_evolvable",
        name="",
        domain="composite genetic architecture",
        default=False,
        info="",
        dtype=bool,
        drange="",
    ),
    "G_neut_agespecific": Parameter(
        key="G_neut_agespecific",
        name="",
        domain="composite genetic architecture",
        default=False,
        info="",
        dtype=bool,
        drange="",
    ),
    "G_neut_interpreter": Parameter(
        key="G_neut_interpreter",
        name="",
        domain="composite genetic architecture",
        default="binary",
        info="",
        dtype=str,
        drange="",
    ),
    "G_neut_initgeno": Parameter(
        key="G_neut_initgeno",
        name="",
        domain="composite genetic architecture",
        default=1,
        info="",
        dtype=float,
        drange="",
    ),
    "G_muta_evolvable": Parameter(
        key="G_muta_evolvable",
        name="",
        domain="composite genetic architecture",
        default=False,
        info="",
        dtype=bool,
        drange="",
    ),
    "G_muta_agespecific": Parameter(
        key="G_muta_agespecific",
        name="",
        domain="composite genetic architecture",
        default=False,
        info="",
        dtype=bool,
        drange="",
    ),
    "G_muta_interpreter": Parameter(
        key="G_muta_interpreter",
        name="",
        domain="composite genetic architecture",
        default="binary",
        info="",
        dtype=str,
        drange="",
    ),
    "G_muta_initgeno": Parameter(
        key="G_muta_initgeno",
        name="",
        domain="composite genetic architecture",
        default=1,
        info="Initial mutation rate",
        dtype=float,
        drange="",
    ),
    "G_grow_evolvable": Parameter(
        key="G_grow_evolvable",
        name="",
        domain="composite genetic architecture",
        default=True,
        info="",
        dtype=bool,
        drange="",
    ),
    "G_grow_agespecific": Parameter(
        key="G_grow_agespecific",
        name="",
        domain="composite genetic architecture",
        default=False,
        info="",
        dtype=bool,
        drange="",
    ),
    "G_grow_interpreter": Parameter(
        key="G_grow_interpreter",
        name="",
        domain="composite genetic architecture",
        default="binary",
        info="",
        dtype=str,
        drange="",
    ),
    "G_grow_initgeno": Parameter(
        key="G_grow_initgeno",
        name="",
        domain="composite genetic architecture",
        default=0.5,
        info="",
        dtype=float,
        drange="",
    ),
    "THRESHOLD": Parameter(
        key="THRESHOLD",
        name="",
        domain="composite genetic architecture",
        default=None,  # 3
        info="",
        dtype=int,
        drange="",
    ),
    #
    #
    # GENETIC ARCHITECTURE (modifying)
    "PHENOMAP_SPECS": Parameter(
        key="PHENOMAP_SPECS",
        name="",
        domain="modifying genetic architecture",
        default=[],
        info="",
        dtype=list,
        drange="",
    ),
    "PHENOMAP": Parameter(
        key="PHENOMAP",
        name="",
        domain="modifying genetic architecture",
        default={},
        info="",
        dtype=dict,
        drange="",
    ),
    "G_grow_initpheno": Parameter(
        key="G_grow_initpheno",
        name="",
        domain="composite genetic architecture",
        default=0.5,
        info="",
        dtype=float,
        drange="",
    ),
    "G_muta_initpheno": Parameter(
        key="G_muta_initpheno",
        name="",
        domain="modifying genetic architecture",
        default=0.001,
        info="Initial mutation rate",
        dtype=float,
        drange="",
    ),
    "G_surv_initpheno": Parameter(
        key="G_surv_initpheno",
        name="",
        domain="modifying genetic architecture",
        default=1,
        info="Initial survival rate",
        dtype=float,
        drange="",
    ),
    "G_repr_initpheno": Parameter(
        key="G_repr_initpheno",
        name="",
        domain="modifying genetic architecture",
        default=1,
        info="Initial fertility rate",
        dtype=float,
        drange="",
    ),
    "G_neut_initpheno": Parameter(
        key="G_neut_initpheno",
        name="",
        domain="modifying genetic architecture",
        default=1,
        info="",
        dtype=float,
        drange="",
    ),
    #
    #
    # OTHER
    "SPECIES_PRESET": Parameter(
        key="SPECIES_PRESET",
        name="",
        domain="other",
        default=None,
        info="",
        dtype=str,
        drange="None or [" + ",".join(PRESET_INFO.keys()) + "]",
        inrange=lambda x: x in PRESET_INFO.keys() or x is None,
        show_in_docs=False,
    ),
    #
    #
    # TIME SCALES
    "STEPS_PER_SIMULATION": Parameter(
        key="STEPS_PER_SIMULATION",
        name="",
        domain="other",
        default=100000,
        info="Number of steps for the simulation to execute",
        dtype=int,
        drange="[1, inf)",
        inrange=lambda x: x >= 1,
        serverrange=lambda x: x <= 1000000,
        serverrange_info="[1,1000000]",
        evalrange=[1, 10000000],
    ),
    "AGE_LIMIT": Parameter(
        key="AGE_LIMIT",
        name="",
        domain="other",
        default=50,
        info="Maximum achievable lifespan (in steps)",
        info_extended="Maximum evolved lifespan is lower than the technically restricted, maximum achievable lifespan.",
        dtype=int,
        drange="[1, inf)",
        inrange=lambda x: x >= 1,
        serverrange=lambda x: x <= 100,
        serverrange_info="[1,100]",
        evalrange=[15, 100],
    ),
    "FRAILTY_MODIFIER": Parameter(
        key="FRAILTY_MODIFIER",
        name="",
        domain="other",
        default=1,
        info="Age-dependent modifier of mortality",
        dtype=float,
        drange="[1, inf)",
        inrange=lambda x: x >= 0,
        serverrange=lambda x: x < 100,
        serverrange_info="[1,100]",
        evalrange=[1, 100],
    ),
    #
    #
    # TECHNICAL
    "MUTATION_METHOD": Parameter(
        key="MUTATION_METHOD",
        name="",
        domain="technical",
        default="by_bit",
        info="Vectorized or non-vectorized method of calculating incidence of new mutations",
        info_extended="Mutate by XOR with a randomized bit matrix ('by_bit') or generate random indices to mutate ('by_index')",
        dtype=str,
        drange="{by_bit, by_index}",
        inrange=lambda x: x in ("by_bit", "by_index"),
    ),
    "RANDOM_SEED": Parameter(
        key="RANDOM_SEED",
        name="",
        domain="technical",
        default=None,
        info="Number used as seed for pseudorandom number generator",
        info_extended="If nothing is given, a random integer will be used as the seed; otherwise the given integer will be used as the seed",
        dtype=int,
        drange="{None, (-inf, inf)}",
        inrange=lambda x: True,
    ),
    "PHENOMAP_METHOD": Parameter(
        key="PHENOMAP_METHOD",
        name="",
        domain="technical",
        default="by_loop",
        info="Non-vectorized, vectorized and blank method of calculating phenotypes from genotypes",
        info_extended="Blank method disables pleiotropy.",
        dtype=str,
        drange="{by_loop, by_dot, by_dummy}",
        inrange=lambda x: x in ("by_loop", "by_dot", "by_dummy"),
    ),
}
