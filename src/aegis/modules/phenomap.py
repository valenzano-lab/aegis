import numpy as np


class Phenomap:
    """Modulator of phenotypic effects
    
    Transforms interpreter values of loci into phenotypic values of traits.
    When no PHENOMAP_SPECS are given, the interpreter values of loci are phenotypic values of traits they are encoding.
    When PHENOMAP_SPECS are given, interpreter values of some loci affect the phenotypic values not only of traits they are
        directly encoding but also of other traits.

    Phenotypic values can loosely be understood as levels of final quantitative traits, and some loci affecting
        multiple traits can be understood as pleiotropy.
    """

    def __init__(self, PHENOMAP_SPECS, gstruc_length):
        """Initialize class.
        
        Arguments:
            PHENOMAP_SPECS: A list of triples (A, B, C) where A is the locus which affects trait B with strength C 
            gstruc_length: The length of genome, i.e. number of loci
        """

        # If no arguments are passed, this class becomes a dummy that does not do anything
        if PHENOMAP_SPECS == []:
            self.dummy = True
        else:
            self.dummy = False
            self.map_ = np.diag([1.0] * gstruc_length)
            for geno_i, pheno_i, weight in PHENOMAP_SPECS:
                self.map_[geno_i, pheno_i] = weight

    def __call__(self, probs):
        """Translate interpreted values into phenotypic values.
        
        Arguments:
            probs: A numpy array of numbers calculated by interpreting genomes using Interpreter
        
        Returns:
            A numpy array of probabilities of events encoded by traits (probabilities to survive, to reproduce, to mutate, ...)
        """
        return probs if self.dummy else np.clip(probs.dot(self.map_), 0, 1)
