import numpy as np


class MatingManager:
    def __init__(self):
        pass

    def pair_up_polygamously(self, sexes, parent_positions=None, max_search_radius=0):
        """
        Return slot indices (into the reproducing pool) of paired males and females.

        When `parent_positions` is None: classical well-mixed pairing — males and
        females are shuffled and paired up to min(n_males, n_females) pairs.

        When `parent_positions` is provided (LATTICE_MODE=True): expanding-ring
        spatial pairing — for each female, search out from her cell up to
        `max_search_radius` hex rings; pair her with a random male found in the
        closest ring that contains any male. Males can mate with multiple
        females (polygamous), so a single male slot may be paired multiple times.
        Females who find no male within the search radius are not paired.
        """
        indices_male = (sexes == 0).nonzero()[0]
        indices_female = (sexes == 1).nonzero()[0]

        if parent_positions is None:
            # Classical well-mixed pairing (unchanged behaviour)
            n_pairs = min(len(indices_male), len(indices_female))
            np.random.shuffle(indices_male)
            np.random.shuffle(indices_female)
            males = indices_male[:n_pairs]
            females = indices_female[:n_pairs]
            return males, females

        # Lattice-aware pairing
        if len(indices_male) == 0 or len(indices_female) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        from aegis_sim.submodels import lattice

        # Build a lookup: (q, r) -> male slot index. If multiple males share a
        # cell (shouldn't happen with one-per-cell, but defensive), the last wins.
        male_pos_to_slot = {}
        for slot in indices_male:
            q, r = parent_positions[slot]
            male_pos_to_slot[(int(q), int(r))] = int(slot)

        paired_males = []
        paired_females = []
        # Shuffle female search order so early females don't monopolise nearby males
        female_order = indices_female.copy()
        np.random.shuffle(female_order)

        for f_slot in female_order:
            f_q, f_r = parent_positions[f_slot]
            found = -1
            for radius in range(1, max_search_radius + 1):
                ring_cells = lattice.ring(int(f_q), int(f_r), radius)
                candidates = []
                for c in ring_cells:
                    key = (int(c[0]), int(c[1]))
                    if key in male_pos_to_slot:
                        candidates.append(male_pos_to_slot[key])
                if candidates:
                    found = int(candidates[np.random.randint(len(candidates))])
                    break
            if found != -1:
                paired_males.append(found)
                paired_females.append(int(f_slot))

        return (
            np.asarray(paired_males, dtype=np.int64),
            np.asarray(paired_females, dtype=np.int64),
        )

    def pair_up_monogamously(self, sexes):
        return
