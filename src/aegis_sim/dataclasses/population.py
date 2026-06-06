import numpy as np
import pickle
import pathlib

from aegis_sim.dataclasses.genomes import Genomes
from aegis_sim.dataclasses.phenotypes import Phenotypes
from aegis_sim import submodels


class Population:
    """Population data

    Contains demographic, genetic and phenotypic data of living individuals.
    """

    attrs = (
        "genomes",
        "ages",
        "births",
        "birthdays",
        "generations",
        "phenotypes",
        "infection",
        "sizes",
        "sexes",
        "ancestry",
        "lineage_id",
        "parent_lineage_id",
        "positions",
    )

    def __init__(
        self,
        genomes: Genomes,
        ages,
        births,
        birthdays,
        phenotypes: Phenotypes,
        infection,
        sizes,
        sexes,
        generations=None,
        ancestry=None,
        lineage_id=None,
        parent_lineage_id=None,
        positions=None,
    ):
        self.genomes = genomes
        self.ages = ages
        self.births = births
        self.birthdays = birthdays
        self.phenotypes = phenotypes
        self.infection = infection
        self.sizes = sizes
        self.sexes = sexes
        self.generations = generations
        # ancestry: bool array, same shape as genomes.array; True = introgressed from source pop.
        # None when introgression tracking is disabled (standard runs).
        self.ancestry = ancestry
        # lineage_id, parent_lineage_id: int64 arrays of length len(genomes).
        # Both None when LINEAGE_TRACING is disabled (standard runs).
        self.lineage_id = lineage_id
        self.parent_lineage_id = parent_lineage_id
        # positions: int32 array of shape (n, 2) holding (q, r) axial hex coordinates
        # on the toroidal lattice. None when LATTICE_MODE is disabled (standard runs).
        self.positions = positions

        assert isinstance(phenotypes, Phenotypes)

        if not (
            len(genomes)
            == len(ages)
            == len(births)
            == len(birthdays)
            == len(phenotypes)
            == len(infection)
            == len(sizes)
            == len(sexes)
            # == len(generations)
        ):
            raise ValueError("Population attributes must have equal length")

    def __setstate__(self, state):
        # Backward compat: pickles saved before ancestry/lineage/positions fields were added
        self.__dict__.update(state)
        if "ancestry" not in self.__dict__:
            self.ancestry = None
        if "lineage_id" not in self.__dict__:
            self.lineage_id = None
        if "parent_lineage_id" not in self.__dict__:
            self.parent_lineage_id = None
        if "positions" not in self.__dict__:
            self.positions = None

    def __len__(self):
        """Return the number of living individuals."""
        return len(self.genomes)

    def __getitem__(self, index):
        """Return a subpopulation."""
        return Population(
            genomes=self.genomes.get(individuals=index),
            ages=self.ages[index],
            births=self.births[index],
            birthdays=self.birthdays[index],
            phenotypes=self.phenotypes.get(individuals=index),
            infection=self.infection[index],
            sizes=self.sizes[index],
            sexes=self.sexes[index],
            generations=self.generations[index] if self.generations is not None else None,
            ancestry=self.ancestry[index] if self.ancestry is not None else None,
            lineage_id=self.lineage_id[index] if self.lineage_id is not None else None,
            parent_lineage_id=self.parent_lineage_id[index] if self.parent_lineage_id is not None else None,
            positions=self.positions[index] if self.positions is not None else None,
        )

    def __imul__(self, index):
        """Redefine itself as its own subpopulation."""
        for attr in self.attrs:
            if attr == "genomes":
                self.genomes.keep(individuals=index)
            elif attr == "phenotypes":
                self.phenotypes.keep(individuals=index)
            elif attr == "generations":
                self.generations = None
            elif attr == "ancestry":
                if self.ancestry is not None:
                    self.ancestry = self.ancestry[index]
            elif attr in ("lineage_id", "parent_lineage_id", "positions"):
                current = getattr(self, attr)
                if current is not None:
                    setattr(self, attr, current[index])
            else:
                setattr(self, attr, getattr(self, attr)[index])
        return self

    def __iadd__(self, population):
        """Merge with another population."""

        for attr in self.attrs:
            if attr == "genomes":
                self.genomes.add(population.genomes)
            elif attr == "phenotypes":
                assert isinstance(population.phenotypes, Phenotypes)
                self.phenotypes.add(population.phenotypes)
            elif attr == "generations":
                self.generations = None
            elif attr == "ancestry":
                if self.ancestry is not None and population.ancestry is not None:
                    self.ancestry = np.concatenate([self.ancestry, population.ancestry])
                elif self.ancestry is not None or population.ancestry is not None:
                    # one side has ancestry tracking, the other doesn't — treat missing as all-native
                    a = self.ancestry if self.ancestry is not None else np.zeros(self.genomes.array.shape, dtype=np.bool_)
                    b = population.ancestry if population.ancestry is not None else np.zeros(population.genomes.array.shape, dtype=np.bool_)
                    self.ancestry = np.concatenate([a, b])
            elif attr in ("lineage_id", "parent_lineage_id"):
                self_val = getattr(self, attr)
                other_val = getattr(population, attr)
                if self_val is None and other_val is None:
                    continue
                # If only one side has lineage IDs, treat the missing side as -1 sentinels.
                # (No lineage tracking on one side means we cannot reconstruct ancestry — keep what we have.)
                if self_val is None:
                    self_val = np.full(len(self.genomes), -1, dtype=np.int64)
                if other_val is None:
                    other_val = np.full(len(population.genomes), -1, dtype=np.int64)
                setattr(self, attr, np.concatenate([self_val, other_val]))
            elif attr == "positions":
                self_val = self.positions
                other_val = population.positions
                if self_val is None and other_val is None:
                    continue
                # Merging spatial + non-spatial populations is unusual. If one side has
                # positions and the other doesn't, sentinel-fill (-1, -1) for the missing side.
                # Lattice submodel must claim cells for these once the merge is realized.
                if self_val is None:
                    self_val = np.full((len(self.genomes), 2), -1, dtype=np.int32)
                if other_val is None:
                    other_val = np.full((len(population.genomes), 2), -1, dtype=np.int32)
                self.positions = np.concatenate([self_val, other_val])
            else:
                val = np.concatenate([getattr(self, attr), getattr(population, attr)])
                setattr(self, attr, val)
        return self

    # def shuffle(self):
    #     order = np.random.arange(len(self))
    #     np.random.shuffle(order)
    #     self *= order

    @staticmethod
    def load_pickle_from(path: pathlib.Path):
        assert path.exists(), f"pickle_path {path} does not exist"
        with open(path, "rb") as file_:
            return pickle.load(file_)

    def save_pickle_to(self, path):
        with open(path, "wb") as file_:
            pickle.dump(self, file_)

    @staticmethod
    def initialize(n, AGE_LIMIT):
        from aegis_sim.parameterization import parametermanager
        from aegis_sim import variables

        genomes = Genomes(submodels.architect.architecture.init_genome_array(n))
        ages = np.random.randint(low=0, high=AGE_LIMIT, size=n, dtype=np.int32)
        births = np.zeros(n, dtype=np.int32)
        birthdays = np.zeros(n, dtype=np.int32)
        # generations = np.zeros(n, dtype=np.int32)
        generations = None

        phenotypes = submodels.architect.__call__(genomes)
        assert isinstance(phenotypes, Phenotypes)

        infection = np.zeros(n, dtype=np.int32)
        sizes = np.zeros(n, dtype=np.float32)
        sexes = submodels.sexsystem.get_sex(n)

        if parametermanager.parameters.LINEAGE_TRACING:
            lineage_id = variables.next_lineage_ids(n)
            parent_lineage_id = np.full(n, -1, dtype=np.int64)
        else:
            lineage_id = None
            parent_lineage_id = None

        # Spatial-model: when LATTICE_MODE is on, assign each individual a unique cell
        # on the toroidal hex lattice. The lattice submodel owns the cell-occupancy
        # bookkeeping; here we just record each individual's (q, r) coords. When
        # LATTICE_MODE is off (default), positions stays None and behavior is unchanged.
        if parametermanager.parameters.LATTICE_MODE:
            positions = submodels.lattice.assign_initial_positions(n)
        else:
            positions = None

        return Population(
            genomes=genomes,
            ages=ages,
            births=births,
            birthdays=birthdays,
            generations=generations,
            phenotypes=phenotypes,
            infection=infection,
            sizes=sizes,
            sexes=sexes,
            ancestry=None,
            lineage_id=lineage_id,
            parent_lineage_id=parent_lineage_id,
            positions=positions,
        )

    @staticmethod
    def make_eggs(
        offspring_genomes: Genomes,
        step,
        offspring_sexes,
        parental_generations,
        offspring_ancestry=None,
        offspring_lineage_id=None,
        offspring_parent_lineage_id=None,
        offspring_positions=None,
    ):
        n = len(offspring_genomes)
        eggs = Population(
            genomes=offspring_genomes,
            ages=np.zeros(n, dtype=np.int32),
            births=np.zeros(n, dtype=np.int32),
            birthdays=np.zeros(n, dtype=np.int32) + step,
            generations=None,
            phenotypes=Phenotypes.init_phenotype_array(n),
            infection=np.zeros(n, dtype=np.int32),
            sizes=np.zeros(n, dtype=np.float32),
            sexes=offspring_sexes,
            ancestry=offspring_ancestry,
            lineage_id=offspring_lineage_id,
            parent_lineage_id=offspring_parent_lineage_id,
            positions=offspring_positions,
        )
        return eggs
