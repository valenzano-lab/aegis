import numpy as np
from aegis.modules.genetics import interpreter, flipmap, phenomap
from aegis.modules.genetics.gstruc import gstruc


def get(genomes):
    """Translate genomes into an array of phenotypes probabilities."""
    # Apply the flipmap

    envgenomes = flipmap.call(genomes)

    # Apply the interpreter functions
    interpretome = np.zeros(shape=(envgenomes.shape[0], envgenomes.shape[2]), dtype=np.float32)
    for trait in gstruc.get_evolvable_traits():
        loci = envgenomes[:, :, trait.slice]  # fetch
        probs = interpreter.call(loci, trait.interpreter)  # interpret
        interpretome[:, trait.slice] += probs  # add back

    # Apply phenomap

    phenotypes = phenomap.call(interpretome)

    # Apply lo and hi bound
    for trait in gstruc.get_evolvable_traits():
        lo, hi = trait.lo, trait.hi
        phenotypes[:, trait.slice] = phenotypes[:, trait.slice] * (hi - lo) + lo

    return phenotypes


def slice_phenotype_trait(phenotypes, trait_name):
    return phenotypes[:, gstruc.get_trait(trait_name).slice]
