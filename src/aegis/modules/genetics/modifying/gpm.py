import logging

from aegis.hermes import hermes


class GPM:
    """
    Order of elements in the vector does not matter.
    """

    def __init__(self, AGE_LIMIT, phenomatrix, phenolist):
        self.AGE_LIMIT = AGE_LIMIT
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
