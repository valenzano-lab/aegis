import numpy as np
from aegis import var
from aegis import cnf


def do(genomes):
    """Return recombined chromatids."""

    if cnf.RECOMBINATION_RATE == 0:
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
    rr = cnf.RECOMBINATION_RATE / 2  # / 2 because you are generating two random vectors (fwd and bkd)
    reco_fwd = (var.rng.random(chromatid1.shape, dtype=np.float32) < rr) * -2 + 1
    reco_bkd = (var.rng.random(chromatid2.shape, dtype=np.float32) < rr) * -2 + 1

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
