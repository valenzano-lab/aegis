import logging

from aegis.hermes import hermes


class GPM:
    """
    Order of elements in the vector does not matter.
    """

    def __init__(self, MAX_LIFESPAN, phenomatrix, phenolist):
        self.MAX_LIFESPAN = MAX_LIFESPAN
        self.phenomatrix = phenomatrix
        self.phenolist = phenolist

        self.dummy = self.phenolist == [] and self.phenomatrix is None
        if self.dummy:
            logging.info("Phenomap inactive")

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
            # phenodiff = np.zeros(shape=(popsize, self.MAX_LIFESPAN * 4))
            for vec_index, trait, age, magnitude in self.phenolist:
                vec_state = vectors[:, vec_index]
                phenotype_change = vec_state * magnitude
                phenotype_index = self.__where(trait, age)
                phenodiff[:, phenotype_index] += phenotype_change
            return phenodiff

        else:
            raise Exception("Neither phenomatrix nor phenolist has been provided.")

    def __where(self, traitname, age=None):
        # Order of traits is hard-encoded and is: surv, repr, muta, neut
        order = {"surv": 0, "repr": 1, "muta": 2, "neut": 3}[traitname]
        if age is None:
            return slice(self.MAX_LIFESPAN * order, self.MAX_LIFESPAN * (order + 1))
        else:
            return self.MAX_LIFESPAN * order + age

    def add_initial_values(self, array):
        array = array.copy()

        for traitname, d in hermes.traits.items():
            where = self.__where(traitname)
            array[:, where] += d.initpheno
        return array

    def __call__(self, interpretome, zeropheno):
        if self.dummy:
            return interpretome
        else:
            phenodiff = self.phenodiff(interpretome, zeropheno=zeropheno)
            phenowhole = self.add_initial_values(phenodiff)
            return phenowhole
