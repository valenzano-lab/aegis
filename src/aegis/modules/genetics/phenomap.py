"""Modulator of phenotypic effects

Transforms interpreter values of loci into phenotypic values of traits.
When no PHENOMAP_SPECS are given, the interpreter values of loci are phenotypic values of traits they are encoding.
When PHENOMAP_SPECS are given, interpreter values of some loci affect the phenotypic values not only of traits they are
    directly encoding but also of other traits.

Phenotypic values can loosely be understood as levels of final quantitative traits, and some loci affecting
    multiple traits can be understood as pleiotropy.
"""

import numpy as np
import pandas as pd
from aegis.modules.genetics import gstruc


class Phenomap:
    def init(self, PHENOMAP_SPECS, PHENOMAP_METHOD):
        self.PHENOMAP_SPECS = PHENOMAP_SPECS
        self.PHENOMAP_METHOD = PHENOMAP_METHOD

        if self.PHENOMAP_SPECS == []:
            self.map_ = None
            self.call = self._by_dummy
        else:
            trios = list(self.unfold_specs())

            self.map_ = np.diag([1.0] * gstruc.get_shape()[1])
            for locus1, locus2, weight in trios:
                self.map_[locus1, locus2] = weight

            self.call = {
                "by_dot": self._by_dot,
                "by_loop": self._by_loop,
            }[self.PHENOMAP_METHOD]

            # Variables for Phenomap.self._by_loop
            _ = np.array(list(zip(*trios)))
            self._by_loop_loc1 = _[0].astype(int)
            self._by_loop_loc2 = _[1].astype(int)
            self._by_loop_weights = _[2]
            self._by_loop_loc_self = self._by_loop_loc1[
                self._by_loop_loc1 == self._by_loop_loc2
            ]  # Loci that affect themselves; i.e. change the baseline weight from 1 to something else

    def _by_dummy(self, probs):
        """A dummy method for calc function"""
        return probs

    def _by_dot(self, probs):
        """A vectorized method for calc function

        Use when map_ is dense.
        """
        return self.clip(probs.dot(self.map_))

    def _by_loop(self, probs):
        """A naive method for calc function

        Use when map_ is sparse.
        Note that it modifies probs in place.
        """

        # List phenotypic differences caused by loci1, scaled by the given phenomap weights
        diffs = probs[:, self._by_loop_loc1] * self._by_loop_weights

        # Override baseline weights
        probs[:, self._by_loop_loc_self] = 0

        # Add back the phenotypic differences caused by loci1 to loci2
        df = pd.DataFrame(diffs.T).groupby(self._by_loop_loc2).sum()
        loc2 = tuple(df.index)
        probs[:, loc2] += df.to_numpy().T

        return self.clip(probs)

    # Fully static methods

    @staticmethod
    def clip(array):
        """Faster version of np.clip(?, 0, 1)"""
        array[array > 1] = 1
        array[array < 0] = 0
        return array

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
            raise ValueError(f"Pattern '{pattern}' contains two values but there is only one target locus")

        return np.linspace(first, last, n)

    def unfold_specs(self):
        for trait1, scope1, trait2, scope2, pattern2 in self.PHENOMAP_SPECS:
            assert (trait1 is None and scope1 is None) or (trait1 is not None and scope2 is not None)

            # If no scope given, whole trait is affected
            if scope2 is None:
                scope2 = f"{gstruc.get_trait(trait2).start + 1}-{gstruc.get_trait(trait2).end}"
                # Note that PHENOMAP_SPECS scope is interpreted as a 1-indexed inclusive interval

            pos2 = gstruc.get_trait(trait2).start
            loci2 = self.decode_scope(scope2) + pos2 - 1  # -1 because the PHENOMAP_SPECS is 1-indexed
            weights = self.decode_pattern(pattern2, len(loci2))

            if trait1 is None:
                loci1 = loci2
            else:
                pos1 = gstruc.get_trait(trait1).start
                loci1 = [scope1 + pos1 - 1] * len(loci2)

            for locus1, locus2, weight in zip(loci1, loci2, weights):
                yield locus1, locus2, weight

    def get_map(self):
        return self.map_


phenomap = Phenomap()
