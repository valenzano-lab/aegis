import numpy as np
from aegis_sim import variables


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
            random_probabilities = variables.rng.random(genomes.shape)

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

        if genomes.size == 0:
            return genomes

        bits_per_genome = genomes[0].size

        muta_prob = self.apply_mutation_age_multiplier(
            mutation_probabilities=muta_prob,
            ages=ages,
            MUTATION_AGE_MULTIPLIER=self.MUTATION_AGE_MULTIPLIER,
        )

        # Calculate number of bits to mutate
        n_mutations_per_individual = variables.rng.binomial(n=bits_per_genome, p=muta_prob, size=len(genomes))
        n_mutations_total = np.sum(n_mutations_per_individual)

        # Generate indices to mutate
        mutation_indices = (
            np.repeat(np.arange(len(genomes)), n_mutations_per_individual),
            variables.rng.integers(genomes.shape[1], size=n_mutations_total),
            variables.rng.integers(genomes.shape[2], size=n_mutations_total),
            variables.rng.integers(genomes.shape[3], size=n_mutations_total),
        )

        # Extract indices of 0-bits and 1-bits
        bits = genomes[mutation_indices]  # NOTE Use tuple for ndarray indexing
        bits0_indices = (~bits).nonzero()[0]
        bits1_indices = bits.nonzero()[0]

        # Take into consideration the MUTATION_RATIO
        bits0_include = variables.rng.random(len(bits0_indices)) < self.rate_0to1
        bits1_include = variables.rng.random(len(bits1_indices)) < self.rate_1to0
        bits0_indices = bits0_indices[bits0_include]
        bits1_indices = bits1_indices[bits1_include]

        # Mutate bits at mutation_indices
        mutation_indices = np.array(mutation_indices)
        genomes[tuple(mutation_indices[:, bits1_indices.T])] = False
        genomes[tuple(mutation_indices[:, bits0_indices.T])] = True

        return genomes

    def _mutate_by_index_packed(self, packed_genomes, muta_prob, ages, n_loci, bits_per_locus):
        """Mutate packed uint8 genomes using XOR masks.

        Same RNG sequence as _mutate_by_index:
        1. apply_mutation_age_multiplier to muta_prob
        2. binomial() for n_mutations_per_individual
        3. integers() for chromatid indices
        4. integers() for locus indices (using n_loci, not n_packed_bytes)
        5. integers() for bit indices (using bits_per_locus)
        6. Read current bit values from packed bytes
        7. random() for MUTATION_RATIO filtering on 0-bits
        8. random() for MUTATION_RATIO filtering on 1-bits
        9. Apply XOR masks to flip bits
        """

        if packed_genomes.size == 0:
            return packed_genomes

        # bits_per_genome must match genomes[0].size in the bool version
        # bool shape is (n, 2, n_loci, bits_per_locus), so genomes[0].size = 2 * n_loci * bits_per_locus
        bits_per_genome = 2 * n_loci * bits_per_locus

        muta_prob = self.apply_mutation_age_multiplier(
            mutation_probabilities=muta_prob,
            ages=ages,
            MUTATION_AGE_MULTIPLIER=self.MUTATION_AGE_MULTIPLIER,
        )

        # RNG call 1: binomial for mutation counts (same as bool version)
        n_mutations_per_individual = variables.rng.binomial(
            n=bits_per_genome, p=muta_prob, size=len(packed_genomes)
        )
        n_mutations_total = np.sum(n_mutations_per_individual)

        # RNG call 2: chromatid indices — shape[1] = 2 for both packed and bool
        ind = np.repeat(np.arange(len(packed_genomes)), n_mutations_per_individual)
        chrom = variables.rng.integers(packed_genomes.shape[1], size=n_mutations_total)

        # RNG call 3: locus indices — use n_loci (same draw as bool version where shape[2] = n_loci)
        locus_indices = variables.rng.integers(n_loci, size=n_mutations_total)

        # RNG call 4: bit indices — use bits_per_locus (same draw as bool version where shape[3] = bpl)
        bit_indices = variables.rng.integers(bits_per_locus, size=n_mutations_total)

        # Convert (locus, bit) to packed coordinates
        flat_bit = locus_indices * bits_per_locus + bit_indices
        byte_idx = flat_bit // 8
        bit_pos = flat_bit % 8

        # Read current bit values from packed array
        current_bits = (packed_genomes[ind, chrom, byte_idx] >> np.uint8(7 - bit_pos)) & np.uint8(1)
        bits_bool = current_bits.astype(np.bool_)

        # Extract indices of 0-bits and 1-bits
        bits0_indices = (~bits_bool).nonzero()[0]
        bits1_indices = bits_bool.nonzero()[0]

        # RNG call 5: MUTATION_RATIO filtering on 0-bits
        bits0_include = variables.rng.random(len(bits0_indices)) < self.rate_0to1
        # RNG call 6: MUTATION_RATIO filtering on 1-bits
        bits1_include = variables.rng.random(len(bits1_indices)) < self.rate_1to0
        bits0_indices = bits0_indices[bits0_include]
        bits1_indices = bits1_indices[bits1_include]

        # Combine surviving mutation indices
        surviving = np.concatenate([bits0_indices, bits1_indices])

        if len(surviving) > 0:
            # Apply XOR masks to flip the targeted bits
            s_ind = ind[surviving]
            s_chrom = chrom[surviving]
            s_byte_idx = byte_idx[surviving]
            s_bit_pos = bit_pos[surviving]
            xor_masks = np.uint8(1 << (7 - s_bit_pos))

            # Apply XOR one at a time to handle duplicate indices correctly
            for i in range(len(surviving)):
                packed_genomes[s_ind[i], s_chrom[i], s_byte_idx[i]] ^= xor_masks[i]

        return packed_genomes

    def _mutate_by_bit_packed(self, packed_genomes, muta_prob, ages, n_loci, bits_per_locus):
        """Fallback: unpack to 4D bool, mutate with existing _mutate_by_bit, repack."""
        n = len(packed_genomes)
        if n == 0:
            return packed_genomes
        ploidy = packed_genomes.shape[1]
        total_bits = n_loci * bits_per_locus
        # Unpack to bool
        unpacked = np.unpackbits(packed_genomes, axis=-1, bitorder='big')[:, :, :total_bits]
        genomes_4d = unpacked.reshape(n, ploidy, n_loci, bits_per_locus).view(np.bool_)
        # Mutate using existing bool logic
        genomes_4d = self._mutate_by_bit(genomes_4d, muta_prob, ages)
        # Repack to uint8
        flat = genomes_4d.reshape(n, ploidy, -1).astype(np.uint8)
        return np.packbits(flat, axis=-1, bitorder='big')

    def _mutate_packed(self, packed_genomes, muta_prob, ages, n_loci, bits_per_locus):
        """Dispatch to packed mutation method based on MUTATION_METHOD."""
        if self.MUTATION_METHOD == "by_index":
            return self._mutate_by_index_packed(packed_genomes, muta_prob, ages, n_loci, bits_per_locus)
        elif self.MUTATION_METHOD == "by_bit":
            return self._mutate_by_bit_packed(packed_genomes, muta_prob, ages, n_loci, bits_per_locus)
        else:
            raise ValueError("MUTATION_METHOD must be 'by_index' or 'by_bit'")


    @staticmethod
    def apply_mutation_age_multiplier(mutation_probabilities, ages, MUTATION_AGE_MULTIPLIER):
        """
        final = initial(1 + age * multiplier)
        """
        multipliers = ages * MUTATION_AGE_MULTIPLIER
        multipliers = multipliers.reshape(mutation_probabilities.shape)
        return mutation_probabilities * (1 + multipliers)


mutator = Mutator()
