from aegis.hermes import hermes


class MatingManager:
    def __init__(self):
        pass

    def pair_up_polygamously(self, sexes):
        """
        Return indices of reproducing, sex-sorted individuals.
        Make sure no same-sex fertilization is happening.
        """

        indices_male = (sexes == 0).nonzero()[0]
        indices_female = (sexes == 1).nonzero()[0]

        # Compute number of pairs
        n_males = len(indices_male)
        n_females = len(indices_female)
        n_pairs = min(n_males, n_females)

        # Shuffle
        hermes.rng.shuffle(indices_male)
        hermes.rng.shuffle(indices_female)

        # Pair up
        males = indices_male[:n_pairs]
        females = indices_female[:n_pairs]

        return males, females

    def pair_up_monogamously(self, sexes):
        return
