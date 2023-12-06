import numpy as np
from aegis import var
from aegis import cnf


rate_0to1 = cnf.MUTATION_RATIO / (1 + cnf.MUTATION_RATIO)
rate_1to0 = 1 / (1 + cnf.MUTATION_RATIO)
_mutate = None  # defined below


def _mutate_by_bit(genomes, muta_prob, random_probabilities=None):
    """Induce germline mutations."""

    if random_probabilities is None:
        random_probabilities = var.rng.random(genomes.shape, dtype=np.float32)

    # Broadcast to fit [individual, chromatid, locus, bit] shape
    mutation_probabilities = muta_prob[:, None, None, None]

    mutate_0to1 = (~genomes) & (
        random_probabilities < (mutation_probabilities * rate_0to1).astype("float32")
    )  # genome == 0 &
    mutate_1to0 = genomes & (
        random_probabilities < (mutation_probabilities * rate_1to0).astype("float32")
    )  # genomes == 1 &

    genomes[mutate_0to1] = 1
    genomes[mutate_1to0] = 0

    return genomes


def _mutate_by_index(genomes, muta_prob):
    """Alternative faster method for introducing mutations.

    Instead of generating a random probability for every bit in the array of genomes,
    generate random indices of bits that could be mutated."""

    bits_per_genome = genomes[0].size

    # Calculate number of bits to mutate
    n_mutations_per_individual = np.random.binomial(n=bits_per_genome, p=muta_prob, size=len(genomes))
    n_mutations_total = np.sum(n_mutations_per_individual)

    # Generate indices to mutate
    mutation_indices = (
        np.repeat(np.arange(len(genomes)), n_mutations_per_individual),
        np.random.randint(genomes.shape[1], size=n_mutations_total),
        np.random.randint(genomes.shape[2], size=n_mutations_total),
        np.random.randint(genomes.shape[3], size=n_mutations_total),
    )

    # Extract indices of 0-bits and 1-bits
    bits = genomes[mutation_indices]  # NOTE Use tuple for ndarray indexing
    bits0_indices = (~bits).nonzero()[0]
    bits1_indices = bits.nonzero()[0]

    # Take into consideration the MUTATION_RATIO
    bits0_include = np.random.random(len(bits0_indices)) < rate_0to1
    bits1_include = np.random.random(len(bits1_indices)) < rate_1to0
    bits0_indices = bits0_indices[bits0_include]
    bits1_indices = bits1_indices[bits1_include]

    # Mutate bits at mutation_indices
    mutation_indices = np.array(mutation_indices)
    genomes[tuple(mutation_indices[:, bits1_indices.T])] = False
    genomes[tuple(mutation_indices[:, bits0_indices.T])] = True

    return genomes


# Set mutation method
if cnf.MUTATION_METHOD == "by_index":
    _mutate = _mutate_by_index
elif cnf.MUTATION_METHOD == "by_bit":
    _mutate = _mutate_by_bit
else:
    raise ValueError("MUTATION_METHOD must be 'by_index' or 'by_bit'")
