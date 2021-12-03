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

    def __init__(self, PHENOMAP_SPECS, gstruc):
        """Initialize class.

        Arguments:
            PHENOMAP_SPECS: A list of triples (A, B, C) where A is the locus which affects trait B with strength C
            gstruc_length: The length of genome, i.e. number of loci
        """

        # If no arguments are passed, this class becomes a dummy that does not do anything
        if PHENOMAP_SPECS == []:
            self.dummy = True
            self.map_ = None
        else:
            self.dummy = False
            self.map_ = np.diag([1.0] * gstruc.length)
            for locus1, locus2, weight in self.unfold_specs(PHENOMAP_SPECS, gstruc):
                self.map_[locus1, locus2] = weight

    def __call__(self, probs):
        """Translate interpreted values into phenotypic values.

        Arguments:
            probs: A numpy array of numbers calculated by interpreting genomes using Interpreter

        Returns:
            A numpy array of probabilities of events encoded by traits (probabilities to survive, to reproduce, to mutate, ...)
        """
        return probs if self.dummy else np.clip(probs.dot(self.map_), 0, 1)

    @staticmethod
    def decode_scope(scope):
        if "," in scope:
            loci = scope.split(",")
        elif "-" in scope:
            from_, to_ = scope.split("-")
            loci = list(range(int(from_), int(to_) + 1))
        else:
            loci = [scope]
        return np.array(loci).astype(int)

    @staticmethod
    def decode_pattern(pattern, n):
        decoded = pattern.split(",")
        first = float(decoded.pop(0))
        last = float(decoded[0]) if decoded else first

        if n == 1 and last != first:
            raise ValueError(
                f"Pattern '{pattern}' contains two values but there is only one target locus"
            )

        return np.linspace(first, last, n)

    @staticmethod
    def unfold_specs(PHENOMAP_SPECS, gstruc):
        for trait1, scope1, trait2, scope2, pattern2 in PHENOMAP_SPECS:

            assert (trait1 is None and scope1 is None) or (
                trait1 is not None and scope2 is not None
            )

            # If no scope given, whole trait is affected
            if scope2 is None:
                scope2 = f"{gstruc[trait2].start + 1}-{gstruc[trait2].end}"  # Note that PHENOMAP_SPECS scope is interpreted as a 1-indexed inclusive interval

            pos2 = gstruc[trait2].start
            loci2 = (
                Phenomap.decode_scope(scope2) + pos2 - 1
            )  # -1 because the PHENOMAP_SPECS is 1-indexed
            weights = Phenomap.decode_pattern(pattern2, len(loci2))

            if trait1 is None:
                loci1 = loci2
            else:
                pos1 = gstruc[trait1].start
                loci1 = [scope1 + pos1 - 1] * len(loci2)

            for locus1, locus2, weight in zip(loci1, loci2, weights):
                yield locus1, locus2, weight
