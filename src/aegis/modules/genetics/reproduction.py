import numpy as np
from aegis.pan import rng
from aegis.modules.genetics.genomes import Genomes


class Reproduction:
    def __init__(self, RECOMBINATION_RATE, REPRODUCTION_MODE, mutator):
        self.RECOMBINATION_RATE = RECOMBINATION_RATE
        self.REPRODUCTION_MODE = REPRODUCTION_MODE
        self.mutator = mutator

    def generate_offspring_genomes(self, genomes, muta_prob):

        if self.REPRODUCTION_MODE == "sexual":
            genomes = self.recombination(genomes)
            genomes, _ = self.assortment(Genomes(genomes))

        genomes = self.mutator._mutate(genomes, muta_prob)
        genomes = Genomes(genomes)
        return genomes

    def recombination(self, genomes):
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
        rr = self.RECOMBINATION_RATE / 2  # / 2 because you are generating two random vectors (fwd and bkd)
        reco_fwd = (rng.random(chromatid1.shape, dtype=np.float32) < rr) * -2 + 1
        reco_bkd = (rng.random(chromatid2.shape, dtype=np.float32) < rr) * -2 + 1

        # Propagate synapse
        reco_fwd_cum = np.cumprod(reco_fwd, axis=1)
        reco_bkd_cum = np.cumprod(reco_bkd[:, ::-1], axis=1)[:, ::-1]

        # Recombine if both sites recombining
        reco_final = (reco_fwd_cum + reco_bkd_cum) == -2

        # Choose bits from first or second chromatid
        # recombined = np.empty(flat_genomes.shape, bool)
        recombined = np.empty(flat_genomes.shape, dtype=np.bool_)
        recombined[:, 0] = np.choose(reco_final, [chromatid1, chromatid2])
        recombined[:, 1] = np.choose(reco_final, [chromatid2, chromatid1])
        recombined = recombined.reshape(genomes.shape)

        return recombined[::2]  # Look at first comment in the function

    @staticmethod
    def _get_order(n_gametes=None, order=None):
        """Return pairings of gametes from different parents."""
        # Extract parent indices twice, and shuffle
        if order is None:
            order = np.repeat(np.arange(n_gametes), 2)
            rng.shuffle(order)

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

    @staticmethod
    def assortment(genomes, order=None):
        """Return assorted chromatids."""

        if order is None:
            order = Reproduction._get_order(n_gametes=len(genomes))

        # Extract gametes
        gametes = genomes.get(individuals=order)

        # Unify gametes
        children = np.empty(genomes.get_array().shape, dtype=np.bool_)
        children[:, 0] = gametes[::2, 0]  # 1st chromatid from 1st parent
        children[:, 1] = gametes[1::2, 1]  # 2nd chromatid from 2nd parent

        return children, order
