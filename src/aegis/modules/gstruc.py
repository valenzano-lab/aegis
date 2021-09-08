
import numpy as np
from aegis.modules.trait import Trait
from aegis.modules.interpreter import Interpreter
from aegis.modules.phenomap import Phenomap
from aegis.modules.environment import Environment
from aegis.panconfiguration import pan


class Gstruc:
    """Genome structure

    Contains information about ploidy, number of loci, and number of bits per locus.
    Calculates phenotypes from input genomes (calls Interpreter, Phenomap and Environment).
    """

    def __init__(self, params, BITS_PER_LOCUS, REPRODUCTION_MODE):

        # Generate traits and save
        self.traits = {}
        self.evolvable = []
        self.length = 0

        for name in Trait.legal:
            trait = Trait(name, params, self.length)
            self.traits[name] = trait
            self.length += trait.length
            if trait.evolvable:
                self.evolvable.append(trait)

        # Infer ploidy
        self.ploidy = {
            "sexual": 2,
            "asexual": 1,
            "asexual_diploid": 2,
        }[REPRODUCTION_MODE]

        self.bits_per_locus = BITS_PER_LOCUS

        self.shape = (self.ploidy, self.length, self.bits_per_locus)

        self.phenomap = Phenomap(
            PHENOMAP_SPECS=params["PHENOMAP_SPECS"],
            gstruc_length=self.length,
        )

        self.interpreter = Interpreter(BITS_PER_LOCUS)

        self.environment = Environment(
            gstruc_shape=self.shape,
            ENVIRONMENT_CHANGE_RATE=params["ENVIRONMENT_CHANGE_RATE"],
        )

    def __getitem__(self, name):
        """Return a Trait instance called {name}."""
        return self.traits[name]

    def initialize_genomes(self, n, headsup=None):
        """Return n initialized genomes.

        Different sections of genome are initialized with a different ratio of ones and zeros
        depending on the G_{}_initial parameter.
        """

        # Initial genomes with a trait.initial fraction of 1's
        genomes = pan.rng.random(size=(n, *self.shape))

        for trait in self.evolvable:
            genomes[:, :, trait.slice] = genomes[:, :, trait.slice] <= trait.initial

        genomes = genomes.astype(bool)

        # Guarantee survival and reproduction values up to a certain age
        if headsup is not None:
            surv_start = self["surv"].start
            repr_start = self["repr"].start
            genomes[:, :, surv_start : surv_start + headsup] = True
            genomes[:, :, repr_start : repr_start + headsup] = True

        return genomes

    def get_phenotype(self, genomes):
        """Translate genomes into an array of phenotypes probabilities."""
        # Apply the environmental map
        envgenomes = self.environment(genomes)

        # Apply the interpreter functions
        interpretome = np.zeros(shape=(envgenomes.shape[0], envgenomes.shape[2]))
        for trait in self.evolvable:
            loci = envgenomes[:, :, trait.slice]  # fetch
            probs = self.interpreter(loci, trait.interpreter)  # interpret
            interpretome[:, trait.slice] += probs  # add back

        # Apply phenomap
        phenotypes = self.phenomap(interpretome)

        # Apply lo and hi bound
        for trait in self.evolvable:
            lo, hi = trait.lo, trait.hi
            phenotypes[:, trait.slice] = phenotypes[:, trait.slice] * (hi - lo) + lo

        return phenotypes
