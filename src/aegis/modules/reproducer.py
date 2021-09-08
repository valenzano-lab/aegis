import numpy as np

from aegis.panconfiguration import pan


class Reproducer:
    """Offspring generator

    Recombines, assorts and mutates genomes of mating individuals to
        create new genomes of their offspring."""

    def __init__(self, RECOMBINATION_RATE, MUTATION_RATIO, REPRODUCTION_MODE):
        self.RECOMBINATION_RATE = RECOMBINATION_RATE
        self.REPRODUCTION_MODE = REPRODUCTION_MODE

        # Mutation rates
        self.rate_0to1 = MUTATION_RATIO / (1 + MUTATION_RATIO)
        self.rate_1to0 = 1 / (1 + MUTATION_RATIO)

    # ================
    # CALLING FUNCTION
    # ================

    def __call__(self, genomes, muta_prob):
        """Exposed method"""

        if self.REPRODUCTION_MODE == "sexual":
            genomes = self._recombine(genomes)
            genomes, _ = self._assort(genomes)

        genomes = self._mutate(genomes, muta_prob)

        return genomes

    # =============
    # RECOMBINATION
    # =============

    def _recombine(self, genomes):
        """Return recombined chromatids."""

        if self.RECOMBINATION_RATE == 0:
            return genomes

        # Recombine two chromatids but pass only one;
        #   thus double the number of chromatids, recobine,
        #   then return only one chromatid from each chromatid pair
        genomes = genomes[np.repeat(np.arange(len(genomes)), 2)]

        # Flatten loci and bits
        flat_genomes = genomes.reshape(len(genomes), 2, -1)

        # Get chromatids
        chromatid1 = flat_genomes[:, 0]
        chromatid2 = flat_genomes[:, 1]

        # Make choice array: when to take recombined and when to take original loci
        # -1 means synapse; +1 means clear
        rr = (
            self.RECOMBINATION_RATE / 2
        )  # / 2 because you are generating two random vectors (fwd and bkd)
        reco_fwd = (pan.rng.random(chromatid1.shape) < rr) * -2 + 1
        reco_bkd = (pan.rng.random(chromatid2.shape) < rr) * -2 + 1

        # Propagate synapse
        reco_fwd_cum = np.cumprod(reco_fwd, axis=1)
        reco_bkd_cum = np.cumprod(reco_bkd[:, ::-1], axis=1)[:, ::-1]

        # Recombine if both sites recombining
        reco_final = (reco_fwd_cum + reco_bkd_cum) == -2

        # Choose bits from first or second chromatid
        # recombined = np.empty(flat_genomes.shape, bool)
        recombined = np.empty(flat_genomes.shape, bool)
        recombined[:, 0] = np.choose(reco_final, [chromatid1, chromatid2])
        recombined[:, 1] = np.choose(reco_final, [chromatid2, chromatid1])
        recombined = recombined.reshape(genomes.shape)

        return recombined[::2]  # Look at first comment in the function

    # ==========
    # ASSORTMENT
    # ==========

    def _get_order(self, n_gametes=None, order=None):
        """Return pairings of gametes from different parents."""
        # Extract parent indices twice, and shuffle
        if order is None:
            order = np.repeat(np.arange(n_gametes), 2)
            pan.rng.shuffle(order)

        # Check for selfing (selfing when pair contains equal parent indices)
        selfed = (order[::2] == order[1::2]).nonzero()[0] * 2

        if len(selfed) == 1:
            # If one parent index pair is selfed,
            #   swap first selfed chromatid with the first chromatid of the previous or next pair
            offset = 2 if selfed[0] == 0 else -2
            order[selfed], order[selfed + offset] = (
                order[selfed + offset],
                order[selfed],
            )
        elif len(selfed) > 1:
            # If multiple parent index pairs are selfed, shift first chromatid of selfed pairs
            order[selfed] = order[np.roll(selfed, 1)]

        return order

    def _assort(self, genomes, order=None):
        """Return assorted chromatids."""

        if order is None:
            order = self._get_order(n_gametes=len(genomes))

        # Extract gametes
        gametes = genomes[order]

        # Unify gametes
        children = np.empty(genomes.shape, bool)
        children[:, 0] = gametes[::2, 0]  # 1st chromatid from 1st parent
        children[:, 1] = gametes[1::2, 1]  # 2nd chromatid from 2nd parent

        return children, order

    # ========
    # MUTATION
    # ========

    def _mutate(self, genomes, muta_prob, random_probabilities=None):
        """Induce germline mutations."""

        if random_probabilities is None:
            random_probabilities = pan.rng.random(genomes.shape)

        # Broadcast to fit [individual, chromatid, locus, bit] shape
        mutation_probabilities = muta_prob[:, None, None, None]

        mutate_0to1 = (genomes == 0) & (
            random_probabilities < (mutation_probabilities * self.rate_0to1)
        )
        mutate_1to0 = (genomes == 1) & (
            random_probabilities < (mutation_probabilities * self.rate_1to0)
        )

        mutate_either = mutate_0to1 + mutate_1to0

        return np.logical_xor(genomes, mutate_either)
