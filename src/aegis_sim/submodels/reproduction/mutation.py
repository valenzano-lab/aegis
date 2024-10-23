import numpy as np


class Mutator:
    def init(self, MUTATION_RATIO, MUTATION_METHOD, MUTATION_AGE_MULTIPLIER):
        self.MUTATION_RATIO = MUTATION_RATIO
        self.MUTATION_METHOD = MUTATION_METHOD
        self.MUTATION_AGE_MULTIPLIER = MUTATION_AGE_MULTIPLIER
        self.rate_0to1 = MUTATION_RATIO / (1 + MUTATION_RATIO)
        self.rate_1to0 = 1 / (1 + MUTATION_RATIO)
        # Set mutation method
        if self.MUTATION_METHOD == "by_index":
            self._mutate = self._mutate_by_index
        elif self.MUTATION_METHOD == "by_bit":
            self._mutate = self._mutate_by_bit
        else:
            raise ValueError("MUTATION_METHOD must be 'by_index' or 'by_bit'")

    def _mutate_by_bit(self, genomes, muta_prob, ages, random_probabilities=None):
        """Induce germline mutations."""

        if random_probabilities is None:
            random_probabilities = np.random.random(genomes.shape)

        # Broadcast to fit [individual, chromatid, locus, bit] shape
        mutation_probabilities = muta_prob[:, None, None, None]

        mutation_probabilities = self.apply_mutation_age_multiplier(
            mutation_probabilities=mutation_probabilities,
            ages=ages,
            MUTATION_AGE_MULTIPLIER=self.MUTATION_AGE_MULTIPLIER,
        )
        mutate_0to1 = (~genomes) & (
            random_probabilities < (mutation_probabilities * self.rate_0to1).astype("float32")
        )  # genome == 0 &
        mutate_1to0 = genomes & (
            random_probabilities < (mutation_probabilities * self.rate_1to0).astype("float32")
        )  # genomes == 1 &

        genomes[mutate_0to1] = 1
        genomes[mutate_1to0] = 0

        return genomes

    def _mutate_by_index(self, genomes, muta_prob, ages):
        """Alternative faster method for introducing mutations.

        Instead of generating a random probability for every bit in the array of genomes,
        generate random indices of bits that could be mutated."""

        bits_per_genome = genomes[0].size

        muta_prob = self.apply_mutation_age_multiplier(
            mutation_probabilities=muta_prob,
            ages=ages,
            MUTATION_AGE_MULTIPLIER=self.MUTATION_AGE_MULTIPLIER,
        )

        # Calculate number of bits to mutate
        n_mutations_per_individual = np.random.binomial(n=bits_per_genome, p=muta_prob, size=len(genomes))
        n_mutations_total = np.sum(n_mutations_per_individual)

        # Generate indices to mutate
        mutation_indices = (
            np.repeat(np.arange(len(genomes)), n_mutations_per_individual),
            np.random.integers(genomes.shape[1], size=n_mutations_total),
            np.random.integers(genomes.shape[2], size=n_mutations_total),
            np.random.integers(genomes.shape[3], size=n_mutations_total),
        )

        # Extract indices of 0-bits and 1-bits
        bits = genomes[mutation_indices]  # NOTE Use tuple for ndarray indexing
        bits0_indices = (~bits).nonzero()[0]
        bits1_indices = bits.nonzero()[0]

        # Take into consideration the MUTATION_RATIO
        bits0_include = np.random.random(len(bits0_indices)) < self.rate_0to1
        bits1_include = np.random.random(len(bits1_indices)) < self.rate_1to0
        bits0_indices = bits0_indices[bits0_include]
        bits1_indices = bits1_indices[bits1_include]

        # Mutate bits at mutation_indices
        mutation_indices = np.array(mutation_indices)
        genomes[tuple(mutation_indices[:, bits1_indices.T])] = False
        genomes[tuple(mutation_indices[:, bits0_indices.T])] = True

        return genomes

    @staticmethod
    def apply_mutation_age_multiplier(mutation_probabilities, ages, MUTATION_AGE_MULTIPLIER):
        """
        final = initial(1 + age * multiplier)
        """
        multipliers = ages * MUTATION_AGE_MULTIPLIER
        multipliers = multipliers.reshape(mutation_probabilities.shape)
        return mutation_probabilities * (1 + multipliers)

mutator = Mutator()