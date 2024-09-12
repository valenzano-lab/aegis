## STARVATION


Starvation is an obligatory source of mortality, useful for modeling death from lack of resources.
The parameter **CARRYING_CAPACITY** specifies the amount of resources.
Generally, each individual requires one unit of resources; otherwise, they are at risk of starvation.
When population size exceeds **CARRYING_CAPACITY**, random individuals will start dying.
The probability to die is genetics-independent (genetics do not confer protection or susceptibility to starvation).
However, age can modify the probability to die, depending on the **STARVATION_RESPONSE**.
When **STARVATION_RESPONSE** is set to treadmill_zoomer, young individuals will start dying first;
for treadmill_boomer, older individuals die first.
Under other **STARVATION_RESPONSE**s, starvation affects all ages equally, but the dynamics of starvation are different.
When response is set to gradual, death from starvation is at first low, but increases with each subsequent
step of insufficient resources (the speed of increase is parameterized by **STARVATION_MAGNITUDE**).
When response is set to treadmill_random, whenever population exceeds the **CARRYING_CAPACITY**, it is immediately
and precisely cut down to **CARRYING_CAPACITY**. In contrast, when response is set to cliff,
whenever **CARRYING_CAPACITY** is exceeded, the population is cut down to a fraction of the **CARRYING_CAPACITY**;
the fraction is specified by the **CLIFF_SURVIVORSHIP** parameter.
Note that if the species is oviparious (**INCUBATION_PERIOD**), the produced eggs do not consume resources and are
immune to starvation mortality (until they hatch).


## PREDATION


Predation is an optional source of mortality, useful for modeling death with prey-predator dynamics.
**PREDATION_RATE** specifies how deadly the predators are; thus if set to 0, no predation deaths will occur.
Apart from **PREDATION_RATE**, the probability that an individual actually gets predated depends also on the number of predators; the response curve is logistic.
The predator population grows according to the logistic Verhulst growth model, whose slope is parameterized by **PREDATOR_GROWTH**.
All individuals are equally susceptible to predation; age and genetics have no impact.


## INFECTION


Infection is an optional source of mortality.
**FATALITY_RATE** specifies how deadly the infection is; thus if set to 0, no deaths from infection will occur.
The infection modeling is inspired by the SIR (susceptible-infectious-removed) model.
Individuals cannot gain immunity, thus can get reinfected many times.
The probability to die from an infection is constant as long as the individual is infected; there is no incubation period nor disease progression.
The same is true for recovering from the disease, which is equal to **RECOVERY_RATE**.
Both of these are independent of age and genetics.
The infectious agent can be transmitted from an individual to an individual but can also be contracted from the environment (and can therefore not be fully eradicated).
The probability to acquire the infection from the environment is equal to **BACKGROUND_INFECTIVITY**, and from other infected individuals it grows with **TRANSMISSIBILITY**
but also (logistically) with the proportion of the infected population.


## ABIOTIC


Abiotic mortality is an optional source of mortality, useful for modeling death by periodic environmental phenomena such as water availability and temperature.
It has no effect when **ABIOTIC_HAZARD_OFFSET** and **ABIOTIC_HAZARD_AMPLITUDE** are set to 0.
It is modeled using periodic functions with a period of **ABIOTIC_HAZARD_PERIOD**, amplitude of **ABIOTIC_HAZARD_AMPLITUDE**,
shape of **ABIOTIC_HAZARD_SHAPE** and constant background mortality of **ABIOTIC_HAZARD_OFFSET** (negative or positive).
Negative hazard is clipped to zero.
Available hazard shapes (waveforms) are flat, sinusoidal, square, triangle, sawtooth, ramp (backward sawtooth) and instant (Dirac comb / impulse train).


## REPRODUCTION


Individuals are fertile starting with **MATURATION_AGE** (can be 0) until **REPRODUCTION_ENDPOINT** (if 0, no REPRODUCTION_ENDPOINT occurs).
Reproduction can be sexual (with diploid genomes) or asexual (with diploid or haploid genomes).
When reproduction is sexual, recombination occurs in gametes at a rate of **RECOMBINATION_RATE**
and gametes will inherit mutations at an age-independent rate
which can be parameterized (genetics-independent) or set to evolve (genetics-dependent).
Mutations cause the offspring genome bit states to flip from 0-to-1 or 1-to-0.
The ratio of 0-to-1 and 1-to-0 can be modified using the **MUTATION_RATIO**.
If the population is oviparous, **INCUBATION_PERIOD** should be set to -1, 1 or greater.
When it is set to -1, all laid eggs hatch only once all living individuals die.
When it is set to 0 or greater, eggs hatch after that specified time.
Thus, when 0, the population has no egg life step.


## RECORDING


AEGIS records a lot of different data.
In brief, AEGIS records
genomic data (population-level allele frequencies and individual-level binary sequences) and
phenotypic data (observed population-level phenotypes and intrinsic individual-level phenotypes),
as well as
derived demographic data (life, death and birth tables),
population genetic data (e.g. effective population size, theta), and
survival analysis data (TE / time-event tables).
Furthermore, it records metadata (e.g. simulation log, processed configuration files) and python pickle files.
Recorded data is distributed in multiple files.
Almost all data are tabular, so each file is a table to which rows are appended as the simulation is running.
The recording rates are frequencies at which rows are added; they are expressed in simulation steps.


## GENETICS


GUI
Every individual carries their own genome. In AEGIS, those are bit strings (arrays of 0's and 1's), passed on from parent to offspring, and mutated in the process.
The submodel genetics transforms genomes into phenotypes; more specifically – into intrinsic phenotypes – biological potentials to exhibit a certain trait (e.g. probability to reproduce).
These potentials are either realized or not realized, depending on the environment (e.g. availability of resources), interaction with other individuals (e.g. availability of mates) and interaction with other traits (e.g. survival).

In AEGIS, genetics is simplified in comparison to the biological reality – it references no real genes and it simulates no molecular interactions; thus, it cannot be used to answer questions about specific genes, metabolic pathways or molecular mechanisms.
However, in AEGIS, in comparison to empirical datasets, genes are fully functionally characterized (in terms of their impact on the phenotype), and are to be studied as functional, heritable genetic elements – in particular, their evolutionary dynamics.

The configuration of genetics – the genetic architecture – is highly flexible. This includes specifying which traits are evolvable number of genetic elements (i.e. size of genome)...
AEGIS offers two genetic architectures – composite and modifying. They are mutually exclusive and are described in detail below...


## COMPOSITE GENETIC ARCHITECTURE


- when pleiotropy is not needed;
- it is quick, easy to analyze, delivers a diversity of phenotypes
- every trait (surv repr muta neut) can be evolvable or not
- if not evolvable, the value is set by !!!
- if evolvable, it can be agespecific or age-independent
- probability of a trait at each age is determined by a BITS_PER_LOCUS adjacent bits forming a "locus" / gene
- the method by which these loci are converted into a phenotypic value is the Interpreter type


## MODIFYING GENETIC ARCHITECTURE


- when pleiotropy is needed
- when all bits are 0, the phenotypic values are the ones set from parameters (baseline set in parameters);
vs composite where it would be 0.
- ... dev still required


## ENVIRONMENTAL DRIFT


Environmental drift is deactivated when **ENVDRIFT_RATE** is 0.
Conceptually, environmental drift simulates long-term environmental change such as climate change, resource depletion, pollution, etc.
The main purpose of environmental drift is to allow the population to keep evolving adaptively.
When the environment does not change, the fitness landscape is static – initially, the population evolves adaptively as it climbs the fitness landscape but once it approaches the fitness peak,
natural selection acts mostly to purify new detrimental mutations. When environmental drift is activated, the fitness landscape changes over time – thus, the population keeps evolving adaptively, following the fitness peak.


## TECHNICAL



## OTHER



