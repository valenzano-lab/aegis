import logging

from aegis.hermes import hermes


class GPM:
    """Genotype-phenotype map

    Order of elements in the vector does not matter. # TODO Explain better

    ### GENOTYPE-PHENOTYPE MAP (GPM) ###
    In AEGIS, every individual carries a genome which encodes an intrinsic phenotype.
    A genome can be converted into an intrinsic phenotype using the genotype-phenotype map (GPM).
    Conceptually, the GPM contains the information on how each site affects the intrinsic phenotype
    of the individual (e.g. the first site decreases fertility by 0.15% at age class 28).

    The GPM can be saved in two formats: a list or a matrix.

    If it is a list, it will be a list of quadruple (4-tuple) with the following structure: `index`, `trait`, `age`, `magnitude`.
    Thus a single quadruple encodes an effect of a single site at the index `index` (e.g. 1)
    on the trait `trait` (e.g. fertility) expressed at the age `age` (e.g. 28). The change to the trait is of magnitude `magnitude` (0.85).
    When a site is pleiotropic, there will be multiple quadruples with the same `index`.
    We distringuish between age-pleiotropy (a single site affecting at least one trait at multiple ages) and trait-pleiotropy (a single site affecting multiple traits).

    If the GPM is encoded as a matrix, it is a 3D matrix where dimensions encode `index`, `trait` and `age`,
    while the matrix values encode the `magnitude`s.

    When most sites are age-pleiotropic and trait-pleiotropic, the optimal encoding format is a matrix.
    When most sites have age-specific and trait-specific effects, the optimal encoding format is a list
    rather than a matrix because the matrix will be very sparse (it will carry a lot of 0's).
    """

    def __init__(self, AGE_LIMIT, phenomatrix, phenolist):
        self.AGE_LIMIT = AGE_LIMIT
        self.phenomatrix = phenomatrix
        self.phenolist = phenolist

        self.dummy = self.phenolist == [] and self.phenomatrix is None
        if self.dummy:
            logging.info("Phenomap inactive.")

    def phenodiff(self, vectors, zeropheno):
        """
        Vectors .. matrix of shape (individual, ?); items are real numbers, chromosomes and loci are resolved
        Phenomatrix ..
        Phenolist .. list of (bit_index, trait, age, magnitude)
        """

        if self.phenomatrix is not None:
            # TODO BUG resolve phenomatrix
            return vectors.dot(self.phenomatrix)

        elif self.phenolist is not None:
            phenodiff = zeropheno.copy()
            for vec_index, trait, age, magnitude in self.phenolist:
                vec_state = vectors[:, vec_index]
                phenotype_change = vec_state * magnitude
                phenotype_index = hermes.traits[trait].start + age
                phenodiff[:, phenotype_index] += phenotype_change
            return phenodiff

        else:
            raise Exception("Neither phenomatrix nor phenolist has been provided.")

    def add_initial_values(self, array):
        array = array.copy()

        for traitname, trait in hermes.traits.items():
            array[:, trait.slice] += trait.initpheno
        return array

    def __call__(self, interpretome, zeropheno):
        if self.dummy:
            return interpretome
        else:
            phenodiff = self.phenodiff(interpretome, zeropheno=zeropheno)
            phenowhole = self.add_initial_values(phenodiff)
            return phenowhole
